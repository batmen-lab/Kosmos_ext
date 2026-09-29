"""The synthetic-data mechanism: a gold-trained teacher labels the supp cells.

The augmentation is a *training-objective* change, not a graph trick:

  * the teacher is the gold-only model (`gears_base` / `mlp_base`);
  * supplementary control cells get that teacher's predicted response;
  * the augmented arms combine a gold loss with that synthetic loss -- the
    ungated one by the signed PPI correction, the gated one by the controller.

These tests pin the two things that used to be absent: the supplementary cells
are labeled by the teacher, and the ungated arm's correction has three distinct
terms (`gold`, `pseudo_gold`, `extension`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from kosmos.ppi.perturbation import (  # noqa: E402
    ARMS,
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
SINGLES = [f"G{i:02d}" for i in range(6)]
COMBOS = [("G00", "G01"), ("G02", "G03"), ("G04", "G05")]
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
    for first, second in COMBOS:
        for _ in range(cells):
            value = rng.normal(0, 0.3, size=len(GENES))
            for gene in (first, second):
                value[GENES.index(gene)] -= 1.5
            rows.append({"condition": f"{first}+{second}", "expression": value})
    frame = pd.DataFrame(rows)
    expression = np.stack(frame.pop("expression").to_numpy())
    return pd.concat([frame, pd.DataFrame(expression, columns=GENES)], axis=1)


def _setup(tmp_path):
    frame = synthetic_screen()
    task = build_task(frame, condition_column="condition", gene_columns=GENES)
    examples, _ = build_examples(frame, task)
    labels = sorted({example.label for example in examples})
    splits = split_examples(
        examples,
        perturbation_splits(
            labels, mode="mixed", test_fraction=0.34, validation_fraction=0.17, seed=0
        ),
    )
    controls = frame[frame["condition"] == CONTROL][GENES].to_numpy(dtype=np.float32)
    go = tmp_path / "go.csv"
    go.write_text(
        "source,target,importance\n"
        + "\n".join(f"{a},{b},0.5" for a in SINGLES for b in SINGLES),
        encoding="utf-8",
    )
    graphs = {
        "G_C": coexpression_graph(controls, name="G_C", threshold=0.3, k=6),
        "G_S": coexpression_graph(controls * 0.9, name="G_S", threshold=0.3, k=6),
        "G_GO": go_graph(GENES, reference=go, k=6),
    }
    config = PerturbationTrainingConfig(
        epochs=2, patience=2, batch_size=16, embedding_dim=8, hidden_dim=8,
        gnn_layers=1, top_k_deg=6, seed=0,
    )
    return splits, task, graphs, controls, config


def test_the_augmented_arms_label_supplementary_cells_with_the_teacher(tmp_path, monkeypatch):
    import kosmos.ppi.perturbation.train as train_module
    from kosmos.ppi.gating import GradientGate

    signed_calls: list[dict] = []
    real_signed = train_module.ppi_signed_correction

    def signed_spy(**kwargs):
        signed_calls.append(kwargs)
        return real_signed(**kwargs)

    monkeypatch.setattr(train_module, "ppi_signed_correction", signed_spy)

    gate_calls: list = []
    real_combine = GradientGate.combine

    def combine_spy(self, gold_loss, synthetic_loss, parameters, **kwargs):
        gate_calls.append(synthetic_loss)
        return real_combine(self, gold_loss, synthetic_loss, parameters, **kwargs)

    monkeypatch.setattr(GradientGate, "combine", combine_spy)

    built: dict = {}
    real_build = train_module.build_synthetic

    def build_spy(**kwargs):
        built.update(kwargs)
        return real_build(**kwargs)

    monkeypatch.setattr(train_module, "build_synthetic", build_spy)

    splits, task, graphs, controls, config = _setup(tmp_path)
    run_perturbation_experiment(
        splits=splits, task=task, graphs=graphs, out_dir=tmp_path / "run",
        config=config, supplementary_controls=controls,
    )

    # 1) the supplementary cells were labeled by a teacher (a predict callable)
    assert built, "the augmented arms must build synthetic labels"
    assert callable(built["predict"])
    assert len(built["controls"]) == len(controls)

    # 2) the ungated arm used the signed correction, with three distinct terms
    assert signed_calls, "the ungated arm must use the signed PPI correction"
    terms = signed_calls[0]
    for key in ("gold", "pseudo_gold", "extension"):
        assert torch.is_tensor(terms[key]), key
    assert float(terms["gold"]) != float(terms["extension"])
    assert float(terms["pseudo_gold"]) != float(terms["extension"])

    # 3) the gated arm gated the *synthetic loss*, not a graph-increment difference
    assert gate_calls and all(torch.is_tensor(loss) for loss in gate_calls)


def test_the_base_arm_never_touches_the_supplementary_rows(tmp_path):
    splits, task, graphs, controls, config = _setup(tmp_path)
    results = run_perturbation_experiment(
        splits=splits, task=task, graphs=graphs, out_dir=tmp_path / "run",
        config=config, supplementary_controls=controls,
    )
    assert set(results["arms"]) == set(ARMS)
    assert all(
        "train_synthetic_loss" not in entry
        for entry in results["arms"]["gears_base"]["history"]
    )
