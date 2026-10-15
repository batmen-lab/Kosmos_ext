"""The perturbation trainer: batch allocation and the two-stage signed PPI arm.

    stage 1   L_true                         Adam(lr)
    stage 2   L_u        (supp, teacher Δ)   Adam(lr × 10)
              −λ·L_pg    (gold, teacher Δ)   Adam(lr ÷ 10)

with 5 stage-1 epochs per stage-2 epoch by default, and supplementary rows paired
one-for-one with the gold batch's perturbations.
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
        epochs=2, patience=2, batch_size=16, hidden_dim=8,
        top_k_deg=6, mlp_hidden_size=16, seed=0,
    )
    return splits, task, graphs, controls, config


def test_the_default_schedule_and_learning_rates():
    config = PerturbationTrainingConfig()
    assert (config.stage1_epochs, config.stage2_epochs) == (5, 1)
    assert config.lambda_lr_multiplier == 10.0
    assert config.stage2_lr_multiplier == 0.1
    assert (config.lambda_init, config.lambda_min, config.lambda_max) == (1.0, 0.0, 2.0)
    assert set(ARMS) == {
        "gears_base", "gears_augmented_ungated", "gears_gated",
        "mlp_base", "mlp_augmented_ungated", "mlp_gated",
    }
    # GEARS' per-epoch halving is off by default (user, 2026-10-14)
    assert config.lr_decay == 1.0 and config.gated_kappa == 1.0


def test_gold_batches_use_every_training_cell_exactly_once(tmp_path):
    from kosmos.ppi.perturbation.train import gold_batches

    splits, task, _, _, config = _setup(tmp_path)
    seen = sum(
        len(batch.labels)
        for batch in gold_batches(splits["train"], task, config, np.random.default_rng(0))
    )
    assert seen == len(splits["train"])


def test_the_supplementary_sampler_exhausts_the_pool_before_repeating():
    from kosmos.ppi.perturbation.train import SupplementarySampler

    pool = np.arange(10, dtype=np.float32).reshape(10, 1)
    sampler = SupplementarySampler(pool, np.random.default_rng(0))
    first = sampler.draw(6)[:, 0].tolist() + sampler.draw(4)[:, 0].tolist()
    assert sorted(first) == list(range(10))
    # a draw larger than what is left wraps into a fresh pass
    assert len(sampler.draw(13)) == 13


def test_stage2_steps_the_model_with_sgd_at_lr_div10_and_lambda_with_adam_at_lr_x10(tmp_path, monkeypatch):
    splits, task, graphs, controls, config = _setup(tmp_path)
    config.arms = ("gears_augmented_ungated",)
    config.epochs, config.stage1_epochs, config.stage2_epochs = 2, 1, 1
    created: list = []
    real_adam, real_sgd = torch.optim.Adam, torch.optim.SGD

    def spy(real, name):
        def make(params, lr=1e-3, **kwargs):
            optimizer = real(params, lr=lr, **kwargs)
            created.append((name, [(g["lr"], g["weight_decay"]) for g in optimizer.param_groups]))
            return optimizer
        return make

    monkeypatch.setattr(torch.optim, "Adam", spy(real_adam, "adam"))
    monkeypatch.setattr(torch.optim, "SGD", spy(real_sgd, "sgd"))
    results = run_perturbation_experiment(
        splits=splits, task=task, graphs=graphs, out_dir=tmp_path / "run",
        config=config, supplementary_controls=controls,
    )
    lr = config.learning_rate
    # teacher Adam, signed stage-1 Adam, stage-2 SGD (model, no weight decay), λ Adam
    assert [name for name, _ in created] == ["adam", "adam", "sgd", "adam"]
    assert created[2][1] == [(pytest.approx(lr / 10), 0)]
    assert created[3][1][0][0] == pytest.approx(lr * 10)
    stage2 = [e for e in results["arms"]["gears_augmented_ungated"]["history"] if e["phase"] == "stage2"]
    assert stage2 and 0.0 < stage2[0]["lambda"] < 2.0 and stage2[0]["lambda"] != 1.0


def test_the_projected_step_does_not_ascend_any_constraint():
    from kosmos.ppi.perturbation.train import project_to_descent

    torch.manual_seed(0)
    for _ in range(200):
        g, a, b = torch.randn(3, 6).unbind(0)
        d = project_to_descent(g, [a, b])
        assert float(d.dot(a)) >= -1e-5 and float(d.dot(b)) >= -1e-5
    # no conflict: unchanged
    g = torch.tensor([1.0, 1.0])
    assert torch.equal(project_to_descent(g, [torch.tensor([1.0, 0.0])]), g)
    # one conflict: the conflicting component is removed
    d = project_to_descent(torch.tensor([1.0, -1.0]), [torch.tensor([0.0, 1.0])])
    assert torch.allclose(d, torch.tensor([1.0, 0.0]))


def test_only_the_gears_arms_decay_their_learning_rate(tmp_path, monkeypatch):
    splits, task, graphs, controls, config = _setup(tmp_path)
    config.arms = ("gears_base", "mlp_base")
    config.epochs = 3
    config.lr_decay = 0.5  # the GEARS schedule, when asked for
    created: list = []
    real = torch.optim.Adam

    def spy(params, lr=1e-3, **kwargs):
        optimizer = real(params, lr=lr, **kwargs)
        created.append(optimizer)
        return optimizer

    monkeypatch.setattr(torch.optim, "Adam", spy)
    run_perturbation_experiment(
        splits=splits, task=task, graphs=graphs, out_dir=tmp_path / "run",
        config=config, supplementary_controls=controls,
    )
    lr = config.learning_rate
    gears, mlp = created
    assert gears.param_groups[0]["lr"] == pytest.approx(lr * 0.5 ** 3)
    assert mlp.param_groups[0]["lr"] == pytest.approx(lr)


def test_learnable_lambda_is_bounded_and_starts_at_its_init():
    from kosmos.ppi.perturbation.train import LearnableLambda

    lam = LearnableLambda(1.0, 0.0, 2.0)
    assert lam.value() == pytest.approx(1.0)
    with torch.no_grad():
        lam.raw.fill_(100.0)
    assert lam.value() == pytest.approx(2.0)
    with pytest.raises(ValueError):
        LearnableLambda(0.0, 0.0, 2.0)


def test_stage2_pairs_supplementary_rows_with_the_gold_batch_perturbations(tmp_path, monkeypatch):
    import kosmos.ppi.perturbation.train as train_module

    splits, task, graphs, controls, config = _setup(tmp_path)
    config.arms = ("mlp_augmented_ungated",)
    config.epochs, config.stage1_epochs, config.stage2_epochs = 2, 1, 1
    config.supplementary_ratio = 2
    calls: list[list[str]] = []
    real = train_module._loss

    def spy(prediction, target, labels, offset, genes, config):
        calls.append(list(labels))
        return real(prediction, target, labels, offset, genes, config)

    monkeypatch.setattr(train_module, "_loss", spy)
    results = run_perturbation_experiment(
        splits=splits, task=task, graphs=graphs, out_dir=tmp_path / "run",
        config=config, supplementary_controls=controls,
    )
    history = results["arms"]["mlp_augmented_ungated"]["history"]
    assert [entry["phase"] for entry in history] == ["stage1", "stage2"]
    n_batches = -(-len(splits["train"]) // config.batch_size)
    # teacher epochs + the signed arm's stage-1 epoch: one loss per batch
    stage1_calls = n_batches * (2 + 1)
    stage2 = calls[stage1_calls:]
    # per batch: L_pg (gold), L_u (supplementary), L_true (gold, the constraint)
    assert len(stage2) == 3 * n_batches
    for pg_labels, u_labels, true_labels in zip(stage2[0::3], stage2[1::3], stage2[2::3]):
        assert u_labels == pg_labels * 2
        assert true_labels == pg_labels


def test_the_finetune_arm_starts_from_the_trained_base(tmp_path, monkeypatch):
    import kosmos.ppi.perturbation.train as train_module

    splits, task, graphs, controls, config = _setup(tmp_path)
    config.arms = ("gears_base", "gears_finetune_ppi")
    config.epochs, config.stage1_epochs, config.stage2_epochs = 2, 1, 1
    config.finetune_epochs, config.finetune_lr_multiplier = 2, 0.1
    seen: dict = {}
    real = train_module._train_arm

    def spy(arm, net, model, *args, **kwargs):
        if arm == "gears_finetune_ppi":
            teacher = kwargs["teacher"].model
            seen["same_start"] = all(
                torch.equal(a, b) for a, b in zip(model.state_dict().values(), teacher.state_dict().values())
            )
            seen["distinct"] = model is not teacher
            seen["augmented"] = net.augmented
            seen["lr"] = kwargs["learning_rate"]
            seen["epochs"] = kwargs["epochs"]
        return real(arm, net, model, *args, **kwargs)

    monkeypatch.setattr(train_module, "_train_arm", spy)
    results = run_perturbation_experiment(
        splits=splits, task=task, graphs=graphs, out_dir=tmp_path / "run",
        config=config, supplementary_controls=controls,
    )
    assert seen == {"same_start": True, "distinct": True, "augmented": False,
                    "lr": pytest.approx(config.learning_rate * 0.1), "epochs": 2}
    phases = [e["phase"] for e in results["arms"]["gears_finetune_ppi"]["history"]]
    assert phases == ["stage1", "stage2"]


def test_the_signed_arm_uses_the_teacher_even_when_base_is_not_reported(tmp_path):
    splits, task, graphs, controls, config = _setup(tmp_path)
    config.arms = ("gears_augmented_ungated",)
    config.epochs, config.stage1_epochs = 2, 1
    results = run_perturbation_experiment(
        splits=splits, task=task, graphs=graphs, out_dir=tmp_path / "run",
        config=config, supplementary_controls=controls,
    )
    assert set(results["arms"]) == {"gears_augmented_ungated"}
    history = results["arms"]["gears_augmented_ungated"]["history"]
    assert any(entry["phase"] == "stage2" for entry in history)
    assert results["arms"]["gears_augmented_ungated"]["best"]["epoch"] >= 1


def test_without_supplementary_cells_the_signed_arm_never_enters_stage2(tmp_path):
    splits, task, graphs, _, config = _setup(tmp_path)
    config.arms = ("mlp_base", "mlp_augmented_ungated")
    config.epochs, config.stage1_epochs = 3, 1
    results = run_perturbation_experiment(
        splits=splits, task=task, graphs=graphs, out_dir=tmp_path / "run",
        config=config, supplementary_controls=None,
    )
    history = results["arms"]["mlp_augmented_ungated"]["history"]
    assert all(entry["phase"] == "stage1" for entry in history)


def test_every_arm_shares_the_gears_loss_and_its_gene_sets(tmp_path, monkeypatch):
    import kosmos.ppi.perturbation.train as train_module

    splits, task, graphs, controls, config = _setup(tmp_path)
    config.epochs, config.stage1_epochs = 2, 1
    seen: list[tuple] = []
    real = train_module.gears_loss

    def spy(prediction, target, perturbations, **kwargs):
        seen.append((id(kwargs.get("genes_by_perturbation")), kwargs.get("direction_lambda")))
        return real(prediction, target, perturbations, **kwargs)

    monkeypatch.setattr(train_module, "gears_loss", spy)
    run_perturbation_experiment(
        splits=splits, task=task, graphs=graphs, out_dir=tmp_path / "run",
        config=config, supplementary_controls=controls,
    )
    assert len(set(seen)) == 1 and seen[0][1] == config.direction_lambda


def test_run_from_tables_threads_supplementary_controls_to_stage2(tmp_path):
    """An unperturbed table in the plan becomes the stage-2 supplementary pool."""
    from kosmos.ppi.perturbation.run import run_perturbation_task

    frame = synthetic_screen()
    control = frame[frame["condition"] == CONTROL]
    gold = pd.concat([control, frame[frame["condition"].isin(SINGLES)]])
    gold_path, supp_path = tmp_path / "gold.csv", tmp_path / "supp.csv"
    gold.to_csv(gold_path, index=False)
    control.drop(columns=["condition"]).to_csv(supp_path, index=False)
    _, _, _, _, config = _setup(tmp_path)
    config.arms = ("mlp_base", "mlp_augmented_ungated")
    config.epochs, config.stage1_epochs = 2, 1

    results = run_perturbation_task(
        gold_path=gold_path,
        supplementary_paths=[supp_path],
        out_dir=tmp_path / "run",
        condition_column="condition",
        go_reference=tmp_path / "go.csv",
        split_mode="mixed",
        test_fraction=0.34,
        validation_fraction=0.17,
        config=config,
    )

    assert results["supplementary"]["control_cells"] > 0
    history = results["arms"]["mlp_augmented_ungated"]["history"]
    assert [entry["phase"] for entry in history] == ["stage1", "stage2"]


def test_control_cells_are_the_gold_control_rows():
    from kosmos.ppi.perturbation.contract import control_cells

    frame = synthetic_screen()
    task = build_task(frame, condition_column="condition", gene_columns=GENES)
    pool = control_cells(frame, task)
    expected = frame[frame["condition"] == CONTROL][GENES].to_numpy(dtype=np.float32)
    assert np.array_equal(pool["global"], expected)


def test_a_gold_row_is_a_real_control_cell_with_target_y_minus_that_cell(tmp_path):
    from kosmos.ppi.perturbation.contract import control_cells
    from kosmos.ppi.perturbation.train import gold_batches

    frame = synthetic_screen()
    splits, task, _, _, config = _setup(tmp_path)
    pool = control_cells(frame, task)
    controls = {tuple(row) for row in pool["global"].tolist()}
    batch = next(gold_batches(splits["train"], task, config, np.random.default_rng(0), pool))
    for row in batch.control.numpy():
        assert tuple(row.tolist()) in controls
    # target = y − x_c; observed (what evaluation scores) = y − control mean
    # = target + (x_c − mean)
    mean = torch.as_tensor(pool["global"].mean(axis=0))
    assert torch.allclose(batch.observed, batch.target + batch.control - mean, atol=1e-5)


def test_the_mean_control_mode_keeps_the_old_input(tmp_path):
    from kosmos.ppi.perturbation.contract import control_cells
    from kosmos.ppi.perturbation.train import gold_batches

    frame = synthetic_screen()
    splits, task, _, _, config = _setup(tmp_path)
    config.sample_controls = False
    batch = next(
        gold_batches(splits["train"], task, config, np.random.default_rng(0), control_cells(frame, task))
    )
    assert torch.equal(batch.target, batch.observed)


def test_gears_delta_depends_on_the_control_cell_only_when_conditioned():
    from kosmos.ppi.perturbation.model import GearsModel, ModelConfig
    from kosmos.ppi.perturbation.model import sparse_adjacency

    n = 6
    edges = torch.tensor([[0, 1], [1, 0]])
    adjacency = sparse_adjacency(edges, torch.ones(2), n)
    index = torch.zeros((2, 1), dtype=torch.long)
    control = torch.stack([torch.zeros(n), torch.ones(n)])
    for conditioned in (True, False):
        torch.manual_seed(0)
        model = GearsModel(ModelConfig(n_genes=n, hidden_dim=4,
                                       condition_on_control=conditioned)).eval()
        _, delta = model(control, index, adjacency, None, adjacency)
        assert (not torch.allclose(delta[0], delta[1])) is conditioned


def test_the_experiment_records_the_control_input(tmp_path):
    from kosmos.ppi.perturbation.contract import control_cells

    frame = synthetic_screen()
    splits, task, graphs, controls, config = _setup(tmp_path)
    config.arms = ("gears_base",)
    config.epochs = 1
    results = run_perturbation_experiment(
        splits=splits, task=task, graphs=graphs, out_dir=tmp_path / "run", config=config,
        supplementary_controls=controls, control_cells=control_cells(frame, task),
    )
    assert results["config"]["sample_controls"] is True
    assert "real control cell" in results["control_input"]


def test_the_loss_genes_are_gears_non_zero_genes_not_a_top_k_set(tmp_path):
    from kosmos.ppi.perturbation.losses import retained_genes
    from kosmos.ppi.perturbation.train import _retained_genes

    cells = np.array([[1.0, 0.0, 2.0, 0.0], [3.0, 0.0, 0.0, 0.0]])
    assert retained_genes({"A": cells})["A"].tolist() == [0, 2]
    splits, task, _, _, config = _setup(tmp_path)
    genes = _retained_genes(splits["train"])
    # the toy screen has no all-zero gene: every gene is in the loss
    assert all(len(index) == task.n_genes > config.top_k_deg for index in genes.values())


def test_the_direction_term_is_measured_from_the_control_mean():
    from kosmos.ppi.perturbation.losses import gears_loss

    # y − ctrl = +0.5 for both rows; Δ̂ = y − x exactly (no error term)
    target = torch.tensor([[1.0], [0.0]])     # y − x, with x = −0.5 and +0.5
    offset = torch.tensor([[-0.5], [0.5]])    # x − ctrl
    exact = gears_loss(target.clone(), target, ["A", "A"], offset=offset)
    assert float(exact.direction) == 0.0
    # Δ̂ = 0: prediction x vs y; row 1 predicts x − ctrl = −0.5 (wrong sign)
    wrong = gears_loss(torch.zeros_like(target), target, ["A", "A"], offset=offset,
                       direction_lambda=1.0)
    assert float(wrong.direction) == pytest.approx(4.0 / 2)


def test_a_constant_prediction_can_never_be_selected():
    from kosmos.ppi.perturbation.metrics import perturbation_metrics
    from kosmos.ppi.perturbation.train import _selection_score

    observed = {"A": np.array([1.0, 0.0, 2.0]), "B": np.array([0.0, 1.0, 0.5])}
    constant = {label: np.array([0.5, 0.5, 1.2]) for label in observed}
    summary = perturbation_metrics(constant, observed, top_k=2).summary
    assert np.isnan(summary["pearson_specific"])
    assert _selection_score(summary, "pearson_specific") == -np.inf
    # the same constant still has a finite (even positive) plain pearson
    assert np.isfinite(summary["pearson"])
    good = perturbation_metrics(observed, observed, top_k=2).summary
    assert summary and _selection_score(good, "pearson_specific") == pytest.approx(1.0)
    assert _selection_score({"mse_deg": 0.2}, "mse_deg") > _selection_score({"mse_deg": 0.3}, "mse_deg")


# -- gated arms (gatedlossdesign.md) ------------------------------------------


def test_a_conflicting_synthetic_gradient_leaves_the_gold_step():
    from kosmos.ppi.perturbation.train import gated_gradient

    gold = torch.tensor([1.0, 0.0])
    final, stats = gated_gradient(gold, torch.tensor([-1.0, 0.5]))
    assert torch.equal(final, gold) and stats["synthetic_weight"] == 0 and stats["synthetic_active"] == 0
    final, stats = gated_gradient(gold, torch.tensor([0.0, 3.0]))  # orthogonal
    assert torch.equal(final, gold)


def test_an_agreeing_synthetic_gradient_is_scaled_by_cosine_and_magnitude():
    from kosmos.ppi.perturbation.train import gated_gradient

    gold = torch.tensor([1.0, 0.0])
    syn = torch.tensor([3.0, 4.0])  # cos 0.6, norm 5 -> c = 1/5
    final, stats = gated_gradient(gold, syn, kappa=1.0)
    assert stats["gradient_cosine"] == pytest.approx(0.6)
    assert stats["synthetic_weight"] == pytest.approx(0.6 * 0.2)
    assert torch.allclose(final, gold + 0.12 * syn)
    # the synthetic part is never longer than kappa x the gold gradient
    assert float((final - gold).norm()) <= float(gold.norm()) + 1e-6
    small = torch.tensor([0.1, 0.0])  # aligned and smaller: c = 1, weight = cos = 1
    final, stats = gated_gradient(gold, small)
    assert stats["synthetic_weight"] == pytest.approx(1.0) and torch.allclose(final, gold + small)


def test_the_gated_arms_train_every_epoch_with_the_gate_logged(tmp_path):
    splits, task, graphs, controls, config = _setup(tmp_path)
    config.arms = ("gears_gated", "mlp_gated")
    config.epochs = 2
    results = run_perturbation_experiment(
        splits=splits, task=task, graphs=graphs, out_dir=tmp_path / "run",
        config=config, supplementary_controls=controls,
    )
    assert set(results["arms"]) == {"gears_gated", "mlp_gated"}
    for arm in ("gears_gated", "mlp_gated"):
        history = results["arms"][arm]["history"]
        assert [entry["phase"] for entry in history] == ["gated", "gated"]
        for entry in history:
            for key in ("train_gold_loss", "train_synthetic_loss", "gradient_cosine",
                        "synthetic_weight", "synthetic_active_fraction", "gold_grad_norm"):
                assert np.isfinite(entry[key])
            assert 0.0 <= entry["synthetic_weight"] <= 1.0
        assert "gating" in results["arms"][arm]
    assert "Gradient gate" in (tmp_path / "run" / "summary.md").read_text()


def test_without_supplementary_cells_the_gated_arm_is_gold_only(tmp_path):
    splits, task, graphs, _, config = _setup(tmp_path)
    config.arms = ("mlp_gated",)
    config.epochs = 1
    results = run_perturbation_experiment(
        splits=splits, task=task, graphs=graphs, out_dir=tmp_path / "run",
        config=config, supplementary_controls=None,
    )
    assert results["arms"]["mlp_gated"]["history"][0]["phase"] == "stage1"
