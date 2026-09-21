"""DeepSeek-designed torch model factories for the PPI path.

The LLM may choose the model architecture (MLP/CNN/Transformer etc.), but the
PPI loss semantics remain owned by the framework:

- Gradient-descent torch modules only (closed-form solvers are rejected because
  their objective cannot be verified as the signed PPI correction);
- run_ppi_experiment keeps the two-stage 4/1 (or configured) PPILoss schedule;
- baseline and PPI share one seeded initialization from the same factory.
"""

import ast
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


MODEL_DESIGN_SCHEMA = {
    "type": "object",
    "properties": {
        "model_factory_code": {
            "type": "string",
            "description": (
                "Complete python code defining `def model_factory(n_features, "
                "n_classes)` returning a torch.nn.Module that outputs logits "
                "of shape (batch, n_classes). Use ONLY torch/torch.nn."
            ),
        },
        "requires_gradient": {
            "type": "boolean",
            "description": "Must be true; only gradient-trained torch models are accepted.",
        },
        "objective_confirmation": {
            "type": "string",
            "description": (
                "Confirm the model will be optimized with the signed PPI "
                "correction (two-stage PPILoss), not a plain supervised loss."
            ),
        },
        "model_reference": {"type": "string"},
        "architecture_notes": {"type": "string"},
    },
    "required": [
        "model_factory_code",
        "requires_gradient",
        "objective_confirmation",
        "model_reference",
        "architecture_notes",
    ],
}


@dataclass
class PPIModelDesign:
    factory: Callable[[int, int], nn.Module]
    code: str
    model_reference: str
    code_reference: str
    architecture_notes: str
    objective_confirmation: str

    def to_dict(self) -> dict:
        return {
            "model_reference": self.model_reference,
            "code_reference": self.code_reference,
            "architecture_notes": self.architecture_notes,
            "objective_confirmation": self.objective_confirmation,
            "model_factory_code": self.code,
        }


def _allowed_imports(code: str) -> Optional[str]:
    """Return disallowed module name if any; None otherwise."""
    allowed_roots = {"torch", "numpy"}
    allowed_from = {"torch", "torch.nn"}
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax error: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in allowed_roots:
                    return alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module not in allowed_from:
                return node.module
    return None


def _validate_and_build(code: str, n_features: int, n_classes: int) -> PPIModelDesign:
    disallowed = _allowed_imports(code)
    if disallowed:
        raise ValueError(f"Model design imports disallowed module: {disallowed}")
    if "def model_factory" not in code:
        raise ValueError("model_factory_code must define `model_factory`")

    namespace: dict[str, Any] = {
        "torch": torch,
        "nn": nn,
        "np": np,
    }
    try:
        exec(compile(code, "<ppi_model_design>", "exec"), namespace)  # noqa: S102
    except Exception as exc:
        raise ValueError(f"model_factory_code failed to compile/execute: {exc}") from exc

    factory = namespace.get("model_factory")
    if not callable(factory):
        raise ValueError("model_factory is not callable")
    try:
        model = factory(n_features, n_classes)
    except Exception as exc:
        raise ValueError(f"model_factory(n_features, n_classes) failed: {exc}") from exc
    if not isinstance(model, nn.Module):
        raise ValueError("model_factory must return a torch.nn.Module")

    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, n_features))
    if tuple(logits.shape) != (2, n_classes):
        raise ValueError(
            f"Model forward must return logits of shape (batch, {n_classes}); got {tuple(logits.shape)}"
        )
    if not torch.isfinite(logits).all():
        raise ValueError("Model logits contain non-finite values on a random probe")
    code_reference = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
    return PPIModelDesign(
        factory=factory,
        code=code,
        model_reference="deepseek-designed",
        code_reference=code_reference,
        architecture_notes="",
        objective_confirmation="",
    )


def design_ppi_model(
    *,
    research_question: str,
    protocol_text: str,
    n_features: int,
    n_classes: int,
    client: Any,
    seed: int = 42,
    task_type: str = "classification",
) -> PPIModelDesign:
    """Ask the model for a torch model factory and validate the PPI contract."""
    regression = str(task_type) == "regression"
    head = (
        "regression model: it returns ONE value per row, trained with squared "
        "error, and the PPI correction is applied to that same per-row error"
        if regression
        else "classification model: it returns class logits, trained with "
        "cross-entropy, and the PPI correction is applied to that same per-row "
        "loss"
    )
    target = (
        "     The model must be a torch.nn.Module returning one value per row:\n"
        "     shape (batch, 1) or (batch,). Squared error, no softmax."
        if regression
        else "     The model must be a torch.nn.Module returning logits "
        "(batch, n_classes)."
    )
    outputs = (
        f"n_outputs = {n_classes}"
        if regression
        else f"n_classes = {n_classes}"
    )
    prompt = f"""Design a small gradient-trained torch {head} for a
cross-donor single-cell experiment that will be trained with a signed PPI loss
correction (PPILoss two-stage: L_true + lambda*(L_external - m*L_pseudo_gold)).

Research question:
{research_question}

Experiment protocol:
{protocol_text}

Dataset:
n_features = {n_features}
{outputs}
seed = {seed}

Rules:
1. Return a complete python snippet defining ONLY:
   def model_factory(n_features, n_classes):
       ... return model
{target}
2. Use ONLY imports from torch/torch.nn/numpy. No data loading, no fitting, no
   optimizer or loss definition inside the factory.
3. Keep the model small enough for CPU training (prefer a linear head or a small
   MLP; deeper nets are allowed only if justified and still CPU-feasible).
4. requires_gradient MUST be true. objective_confirmation MUST state that the
   model will be optimized under the signed PPI correction with the framework's
   two-stage schedule.
"""
    try:
        response = client.generate_structured(
            prompt=prompt,
            schema=MODEL_DESIGN_SCHEMA,
            max_tokens=6000,
            temperature=0.2,
        )
    except Exception as exc:
        raise ValueError(f"DeepSeek model design request failed: {exc}") from exc

    code = (response or {}).get("model_factory_code", "")
    if not response.get("requires_gradient", False):
        raise ValueError(
            "Rejected model design: only gradient-trained torch models are "
            "allowed in the PPI path."
        )
    objective = (response.get("objective_confirmation") or "").lower()
    if "ppi" not in objective and "correction" not in objective:
        raise ValueError(
            "Rejected model design: objective_confirmation must reference the "
            "signed PPI correction."
        )

    design = _validate_and_build(code, n_features, n_classes)
    design.model_reference = response.get("model_reference") or design.model_reference
    design.architecture_notes = response.get("architecture_notes") or ""
    design.objective_confirmation = response.get("objective_confirmation") or ""
    return design


def write_design(design: PPIModelDesign, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "ppi_model_factory.py"
    path.write_text(
        f'"""DeepSeek-designed PPI model factory. code_reference={design.code_reference}"""\n\n'
        + design.code,
        encoding="utf-8",
    )
    return path
