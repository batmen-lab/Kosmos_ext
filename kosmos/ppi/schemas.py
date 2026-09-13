"""Explicit data roles and serializable classification experiment configuration."""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


def matrix(X, ids, names):
    X = np.asarray(X, dtype=np.float32)
    ids = np.asarray(ids, dtype=str)
    if X.ndim != 2 or not X.shape[1] or not np.isfinite(X).all():
        raise ValueError("X must be a finite numeric matrix with at least one feature")
    if ids.shape != (len(X),) or len(set(ids)) != len(ids) or np.any(ids == ""):
        raise ValueError("sample_ids must be unique nonempty identifiers aligned to X")
    if names is not None and (len(names) != X.shape[1] or len(set(names)) != len(names)):
        raise ValueError("feature_names must uniquely identify every column in order")
    return X.copy(), ids.copy()


@dataclass
class GoldDataset:
    X: np.ndarray
    y: np.ndarray
    sample_ids: np.ndarray
    role: Literal["gold_train", "gold_validation", "final_test"]
    dataset_id: str
    feature_names: list[str] | None = None
    groups: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    representation: Literal["observed"] = "observed"

    def __post_init__(self):
        self.X, self.sample_ids = matrix(self.X, self.sample_ids, self.feature_names)
        self.y = np.asarray(self.y)
        if not len(self.X) or self.y.shape != (len(self.X),):
            raise ValueError("Gold data must have one true label per observation and be nonempty")
        if self.y.dtype.kind not in "biufUS" or (
            self.y.dtype.kind in "f" and not np.isfinite(self.y).all()
        ):
            raise ValueError("Labels must be finite numbers or strings")
        if self.role not in {"gold_train", "gold_validation", "final_test"}:
            raise ValueError("Invalid gold role")
        if not self.dataset_id or self.representation != "observed":
            raise ValueError("Gold must be trusted observed data with a dataset identifier")
        if (
            self.metadata.get("evidence_route")
            or self.metadata.get("representation", "observed") != "observed"
        ):
            raise ValueError("External/translated evidence cannot enter GoldDataset")
        if self.groups is not None:
            self.groups = np.asarray(self.groups, dtype=str)
            if self.groups.shape != (len(self.X),) or np.any(self.groups == ""):
                raise ValueError("Group identifiers must align with observations")


@dataclass
class ExternalEvidenceDataset:
    X: np.ndarray
    sample_ids: np.ndarray
    evidence_route: Literal["DIRECT_PPI", "TRANSLATE_PPI"]
    source_dataset_id: str
    feature_names: list[str] | None = None
    alpha: np.ndarray | None = None
    # Absolute upstream mass, distinct from alpha. Never normalized back to one.
    evidence_weight: np.ndarray | None = None
    groups: np.ndarray | None = None
    audit_metadata: dict[str, Any] = field(default_factory=dict)
    source_labels: np.ndarray | None = None
    role: Literal["external_evidence"] = "external_evidence"

    def __post_init__(self):
        self.X, self.sample_ids = matrix(self.X, self.sample_ids, self.feature_names)
        if self.role != "external_evidence" or self.evidence_route not in {
            "DIRECT_PPI",
            "TRANSLATE_PPI",
        }:
            raise ValueError("External evidence requires an external role and accepted route")
        if not self.source_dataset_id:
            raise ValueError("Evidence requires a source dataset ID")
        for name in ("alpha", "evidence_weight"):
            values = getattr(self, name)
            if values is not None:
                values = np.asarray(values, dtype=float)
                if (
                    values.shape != (len(self.X),)
                    or not np.isfinite(values).all()
                    or (values < 0).any()
                    or (values > 1).any()
                ):
                    raise ValueError(f"{name} must contain aligned values in [0,1]")
                setattr(self, name, values)
        for name in ("groups", "source_labels"):
            values = getattr(self, name)
            if values is not None:
                values = np.asarray(values)
                if values.shape != (len(self.X),):
                    raise ValueError(f"{name} must align with observations")
                setattr(self, name, values)


@dataclass
class PseudoLabeledGold:
    X: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray | None
    sample_ids: np.ndarray


@dataclass
class PseudoLabeledExternal:
    X: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray | None
    sample_ids: np.ndarray
    evidence_metadata: dict[str, Any]
    weights: np.ndarray


class PPITrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    task_type: Literal["classification"] = "classification"
    seed: int = 42
    gold_batch_size: int = Field(default=64, gt=0)
    external_batch_size: int = Field(default=128, gt=0)
    learning_rate: float = Field(default=0.001, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    max_epochs: int = Field(default=50, gt=0)
    patience: int = Field(default=10, gt=0)
    min_delta: float = Field(default=0, ge=0)
    pseudo_mode: Literal["cross_fit", "in_sample", "pretrained"] = "cross_fit"
    cross_fit_folds: int = Field(default=5, ge=2)
    pseudo_targets: Literal["hard", "soft"] = "hard"
    use_evidence_weights: bool = False
    external_weight_budget: float = Field(default=0.5, ge=0, le=1)
    max_external_samples: int = Field(default=10000, ge=0)
    max_rows_per_dataset: int = Field(default=10000, gt=0)
    ppi_lambda: float = Field(default=1, ge=0, le=1)
    schedule: Literal["two_stage", "joint"] = "two_stage"
    stage1_epochs: int = Field(default=4, gt=0)
    stage2_epochs: int = Field(default=1, gt=0)
    stage2_lr_multiplier: float = Field(default=0.1, gt=0)
    label_smoothing: float = Field(default=0, ge=0, lt=1)
    evaluation_metric: Literal[
        "accuracy", "balanced_accuracy", "macro_f1", "macro_auroc", "macro_auprc"
    ] = "balanced_accuracy"
    model_reference: str = "torch-linear-v1"


@dataclass
class PPITrainingResult:
    model: Any
    baseline_model: Any
    classes: np.ndarray
    trained_model_path: str
    baseline_model_path: str
    training_history: dict[str, list[dict]]
    validation_metrics: dict[str, Any]
    baseline_metrics: dict[str, Any]
    delta_metrics: dict[str, float | None]
    n_gold: int
    n_external_available: int
    n_external_used: int
    evidence_sources_used: list[str]
    reproducibility: dict[str, Any]
    selected_epochs: dict[str, int]
    pseudo_gold: PseudoLabeledGold
    pseudo_external: list[PseudoLabeledExternal]

    def summary(self):
        excluded = {"model", "baseline_model", "classes", "pseudo_gold", "pseudo_external"}
        return {
            **{k: v for k, v in vars(self).items() if k not in excluded},
            "classes": self.classes.tolist(),
        }
