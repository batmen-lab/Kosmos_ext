"""The graph-free MLP baselines: Base MLP and teacher-augmented MLP.

They are additive: a caller that does not ask for them still gets exactly the
three GEARS arms, and the augmented arm reduces to the gold-only update when
there is no supplementary evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from kosmos.ppi.perturbation import (  # noqa: E402
    ARMS,
    MLP_ARMS,
    PerturbationTrainingConfig,
    build_examples,
    build_task,
    coexpression_graph,
    go_graph,
    perturbation_splits,
    run_perturbation_experiment,
    split_examples,
)

GENES = [f"G{i:02d}" for i in range(16)]
SINGLES = [f"G{i:02d}" for i in range(4)]
CONTROL = "ctrl"


def synthetic_screen(cells: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(cells):
        base = rng.normal(0, 0.3, size=len(GENES))
        base[0::2] += 0.8
        rows.append({"condition": CONTROL, "expression": base})
    for index, gene in enumerate(SINGLES):
        for _ in range(cells):
            value = rng.normal(0, 0.3, size=len(GENES))
            value[index] -= 1.5
            value[index % 2 :: 2] -= 0.5
            rows.append({"condition": gene, "expression": value})
    for first, second in (("G00", "G01"), ("G02", "G03")):
        for _ in range(cells):
            value = rng.normal(0, 0.3, size=len(GENES))
            for gene in (first, second):
                value[GENES.index(gene)] -= 1.5
            rows.append({"condition": f"{first}+{second}", "expression": value})
    frame = pd.DataFrame(rows)
    expression = np.stack(frame.pop("expression").to_numpy())
    return pd.concat([frame, pd.DataFrame(expression, columns=GENES)], axis=1)


def _go_graph(tmp_path: Path) -> Path:
    path = tmp_path / "go.csv"
    lines = ["source,target,importance"]
    for gene in SINGLES:
        for other in SINGLES:
            lines.append(f"{gene},{other},{1.0 if gene == other else 0.5}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _setup(tmp_path: Path):
    frame = synthetic_screen()
    task = build_task(frame, condition_column="condition", gene_columns=GENES)
    examples, _ = build_examples(frame, task)
    labels = sorted({example.label for example in examples})
    splits_by_label = perturbation_splits(
        labels, mode="mixed", test_fraction=0.34, validation_fraction=0.17, seed=0
    )
    splits = split_examples(examples, splits_by_label)
    controls = frame[frame["condition"] == CONTROL][GENES].to_numpy(dtype=np.float32)
    graphs = {
        "G_C": coexpression_graph(controls, name="G_C", threshold=0.3, k=6),
        "G_S": coexpression_graph(controls * 0.9, name="G_S", threshold=0.3, k=6),
        "G_GO": go_graph(GENES, reference=_go_graph(tmp_path), k=6),
    }
    config = PerturbationTrainingConfig(
        epochs=2, patience=2, batch_size=16, embedding_dim=8, hidden_dim=8,
        gnn_layers=1, top_k_deg=6, mlp_hidden_size=16, seed=0,
        include_mlp_baselines=True,
    )
    return splits, task, graphs, controls, config


def test_the_mlp_baselines_run_alongside_the_gears_arms(tmp_path):
    splits, task, graphs, controls, config = _setup(tmp_path)
    run_dir = tmp_path / "run"

    results = run_perturbation_experiment(
        splits=splits, task=task, graphs=graphs, out_dir=run_dir,
        config=config, supplementary_controls=controls * 0.9,
    )

    arms = set(results["arms"])
    assert set(ARMS) <= arms
    assert set(MLP_ARMS) <= arms
    for arm in MLP_ARMS:
        summary = results["arms"][arm]["test"]["summary"]
        for key in ("mse", "mse_deg", "pearson", "direction_accuracy", "top_k_overlap"):
            assert key in summary
    # Checkpoints and the synthetic-label record the design asks for.
    assert (run_dir / "mlp_base.pt").exists()
    assert (run_dir / "mlp_augmented.pt").exists()
    assert (run_dir / "teacher_checkpoint.pt").exists()
    metadata = json.loads((run_dir / "synthetic_label_metadata.json").read_text())
    # One objective across every arm: the MLP trains on the GEARS loss (Δ), and
    # the supplementary targets are the teacher's labels, never the source's.
    assert metadata["objective"] == "gears"
    assert metadata["target"] == "delta"
    assert metadata["label_source"].startswith("teacher-generated")
    assert metadata["direction_lambda"] == config.direction_lambda
    assert metadata["n_rows"] == len(controls)
    assert results["config"]["mlp_objective"] == "gears"
    # The augmented arm actually used the gate.
    assert any("gate" in entry for entry in results["arms"]["mlp_augmented"]["history"])
    # The two MLP arms share one architecture: same parameter shapes.
    base = results["arms"]["mlp_base"]["test"]["per_perturbation"]
    augmented = results["arms"]["mlp_augmented"]["test"]["per_perturbation"]
    assert set(base) == set(augmented)


def test_the_mlp_baselines_are_off_by_default(tmp_path):
    splits, task, graphs, controls, config = _setup(tmp_path)
    config.include_mlp_baselines = False

    results = run_perturbation_experiment(
        splits=splits, task=task, graphs=graphs, out_dir=tmp_path / "run", config=config
    )

    assert set(results["arms"]) == set(ARMS)
    assert not (tmp_path / "run" / "mlp_base.pt").exists()


def test_the_augmented_mlp_reduces_to_gold_without_supplementary(tmp_path):
    splits, task, graphs, _, config = _setup(tmp_path)

    results = run_perturbation_experiment(
        splits=splits, task=task, graphs=graphs, out_dir=tmp_path / "run",
        config=config, supplementary_controls=None,
    )

    history = results["arms"]["mlp_augmented"]["history"]
    assert history and all("gate" not in entry for entry in history)


def test_the_mlp_model_uses_no_graph(tmp_path):
    from kosmos.ppi.perturbation.mlp import MlpModel, MlpConfig

    model = MlpModel(MlpConfig(n_genes=len(GENES)))
    # Two encoders added, one linear head: no GNN, no adjacency parameter.
    parameter_names = {name for name, _ in model.named_parameters()}
    assert not any("gnn" in name or "adjacency" in name for name in parameter_names)
    control = torch.zeros(2, len(GENES))
    indicator = torch.zeros(2, len(GENES))
    indicator[0, 0] = 1.0
    out = model(control, indicator)
    assert out.shape == (2, len(GENES))


def test_run_from_tables_threads_supplementary_controls_to_the_mlp(tmp_path):
    """The autoresearch entry hands the supplementary control cells to the MLP."""
    from kosmos.ppi.perturbation.run import run_perturbation_task

    frame = synthetic_screen()
    control = frame[frame["condition"] == CONTROL]
    gold = pd.concat([control, frame[frame["condition"].isin(SINGLES)]])
    gold_path, supp_path = tmp_path / "gold.csv", tmp_path / "supp.csv"
    gold.to_csv(gold_path, index=False)
    # an unperturbed atlas: no `condition` column, so every cell is a control
    control.drop(columns=["condition"]).to_csv(supp_path, index=False)

    results = run_perturbation_task(
        gold_path=gold_path,
        supplementary_paths=[supp_path],
        out_dir=tmp_path / "run",
        condition_column="condition",
        go_reference=_go_graph(tmp_path),
        split_mode="mixed",
        test_fraction=0.34,
        validation_fraction=0.17,
        config=PerturbationTrainingConfig(
            epochs=2, patience=2, batch_size=16, embedding_dim=8, hidden_dim=8,
            gnn_layers=1, top_k_deg=6, mlp_hidden_size=16, seed=0,
            include_mlp_baselines=True,
        ),
    )

    assert {"mlp_base", "mlp_augmented"} <= set(results["arms"])
    assert (tmp_path / "run" / "synthetic_label_metadata.json").exists()
    assert any("gate" in entry for entry in results["arms"]["mlp_augmented"]["history"])


def test_the_mlp_objective_defaults_to_the_gears_loss():
    """Architecture comparisons are only valid on a shared training signal."""
    config = PerturbationTrainingConfig()
    assert config.mlp_objective == "gears"
    from kosmos.ppi.perturbation.mlp import _target_kind

    assert _target_kind(config) == "delta"
    config.mlp_objective = "mse"
    assert _target_kind(config) == "expression"


def test_the_mlp_arm_shares_the_gears_loss_and_deg_map(tmp_path, monkeypatch):
    """The crux: the MLP arms must optimise the GEARS objective, DEG map and all."""
    import kosmos.ppi.perturbation.mlp as mlp
    from kosmos.ppi.perturbation.mlp import run_mlp_arms
    from kosmos.ppi.perturbation.train import _deg_map

    splits, task, graphs, controls, config = _setup(tmp_path)
    deg_map = _deg_map(splits["train"], config.top_k_deg)
    seen: dict = {}
    real = mlp.gears_loss

    def spy(prediction, target, perturbations, **kwargs):
        seen["deg_by_perturbation"] = kwargs.get("deg_by_perturbation")
        seen["direction_lambda"] = kwargs.get("direction_lambda")
        return real(prediction, target, perturbations, **kwargs)

    monkeypatch.setattr(mlp, "gears_loss", spy)

    run_mlp_arms(
        train_examples=splits["train"],
        validation_examples=splits["validation"],
        test_examples=splits["test"],
        task=task,
        config=config,
        out_dir=tmp_path / "run",
        supplementary_controls=controls * 0.9,
        deg_by_perturbation=deg_map,
    )

    assert seen.get("deg_by_perturbation") is deg_map
    assert seen.get("direction_lambda") == config.direction_lambda
