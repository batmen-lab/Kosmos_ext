"""The perturbation backend end to end: contract -> graphs -> three arms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from kosmos.ppi.perturbation import (  # noqa: E402
    ARMS,
    ModelConfig,
    PerturbationTrainingConfig,
    build_examples,
    build_task,
    coexpression_graph,
    go_graph,
    parse_condition,
    perturbation_splits,
    run_perturbation_experiment,
    split_examples,
    write_contract,
)
from kosmos.ppi.perturbation.model import GearsModel, sparse_adjacency  # noqa: E402

GENES = [f"G{i:02d}" for i in range(24)]
SINGLES = [f"G{i:02d}" for i in range(6)]
COMBOS = [("G00", "G01"), ("G02", "G03"), ("G04", "G05")]
CONTROL = "ctrl"


def synthetic_screen(cells_per_condition: int = 12, seed: int = 0) -> pd.DataFrame:
    """A structured perturbation screen: a hit moves its own module."""
    rng = np.random.default_rng(seed)
    rows = []
    for gene_index, gene in enumerate(GENES):
        module = gene_index % 4
        base = rng.normal(0.0, 0.3, size=len(GENES))
        base[module::4] += 0.8  # genes in a module covary in controls
        for _ in range(cells_per_condition):
            rows.append(
                {
                    "condition": CONTROL,
                    "cell_type": "ct",
                    "expression": base + rng.normal(0, 0.1, size=len(GENES)),
                }
            )
    for gene in SINGLES:
        index = GENES.index(gene)
        module = index % 4
        for _ in range(cells_per_condition):
            value = rng.normal(0.0, 0.3, size=len(GENES))
            value[module::4] += 0.8
            value[index] -= 1.5
            value[module::4] -= 0.6  # knock a gene down, its module follows
            rows.append({"condition": gene, "cell_type": "ct", "expression": value})
    for first, second in COMBOS:
        for _ in range(cells_per_condition):
            value = rng.normal(0.0, 0.3, size=len(GENES))
            for gene in (first, second):
                value[GENES.index(gene)] -= 1.5
            rows.append({"condition": f"{first}+{second}", "cell_type": "ct", "expression": value})
    frame = pd.DataFrame(rows)
    expression = np.stack(frame.pop("expression").to_numpy())
    frame = pd.concat([frame, pd.DataFrame(expression, columns=GENES)], axis=1)
    return frame


def small_go_graph(tmp_path):
    """A stand-in for the real GO file: the six perturbed genes are similar."""
    path = tmp_path / "go.csv"
    lines = ["source,target,importance"]
    for gene in SINGLES:
        for other in SINGLES:
            lines.append(f"{gene},{other},{1.0 if gene == other else 0.5}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_a_condition_string_is_parsed_into_its_genes():
    assert parse_condition("CBL") == ("CBL",)
    assert parse_condition("CBL+CNN1") == ("CBL", "CNN1")
    assert parse_condition("CBL_CNN1") == ("CBL", "CNN1")
    assert parse_condition("ARMCX5-GPRASP2") == ("ARMCX5-GPRASP2",)


def test_the_contract_defines_deltas_against_context_matched_controls():
    frame = synthetic_screen()
    task = build_task(frame, condition_column="condition", gene_columns=GENES)
    examples, report = build_examples(frame, task, context_columns=["cell_type"])

    assert report["control_cells"] == 12 * len(GENES)
    single = next(example for example in examples if example.label == "G00")
    # the knocked-out gene went down, so its delta is negative
    assert single.delta[0] < -0.5
    # and the module around it moved too
    assert single.delta[4] < 0
    assert single.control_group.startswith("ct")


def test_splits_are_by_perturbation_and_cover_unseen_singles_and_combinations():
    labels = SINGLES + ["+".join(combo) for combo in COMBOS]
    mixed = perturbation_splits(labels, mode="mixed", test_fraction=0.34, validation_fraction=0.17, seed=0)
    assert mixed["splits_by"] == "perturbation"
    overlap = set(mixed["train"]) & (set(mixed["test"]) | set(mixed["validation"]))
    assert not overlap, "a perturbation may not be in two splits"
    assert any("+" in label for label in mixed["test"]) or any("+" in label for label in mixed["validation"])

    unseen_single = perturbation_splits(labels, mode="unseen_single", test_fraction=0.34, seed=0)
    assert all("+" not in label for label in unseen_single["test"])
    unseen_combo = perturbation_splits(labels, mode="unseen_combination", test_fraction=0.34, seed=0)
    assert all("+" in label for label in unseen_combo["test"])


def test_the_splits_assign_every_example_once():
    frame = synthetic_screen()
    task = build_task(frame, condition_column="condition", gene_columns=GENES)
    examples, _ = build_examples(frame, task)
    labels = sorted({example.label for example in examples})
    splits = perturbation_splits(labels, mode="mixed", seed=1)
    assigned = split_examples(examples, splits)
    total = sum(len(values) for values in assigned.values())
    assert total == len(examples)


def test_graphs_carry_their_statistics_and_overlap(tmp_path):
    frame = synthetic_screen()
    task = build_task(frame, condition_column="condition", gene_columns=GENES)
    controls = frame[frame["condition"] == CONTROL][GENES].to_numpy(dtype=np.float32)
    gold = coexpression_graph(controls, name="G_C", threshold=0.3, k=6)
    supplementary = coexpression_graph(controls * 1.2, name="G_S", threshold=0.3, k=6)
    go = go_graph(GENES, reference=small_go_graph(tmp_path), k=6)

    assert gold.n_edges > 0 and go.n_edges > 0
    assert gold.stats["kind"] == "coexpression" and go.stats["kind"] == "go_similarity"
    assert 0.0 <= gold.stats["density"] <= 1.0
    from kosmos.ppi.perturbation import edge_overlap

    overlap = edge_overlap(gold, supplementary)
    assert overlap["shared"] > 0 and overlap["jaccard"] > 0


def test_the_model_consumes_control_and_perturbation_and_predicts_a_vector():
    model = GearsModel(ModelConfig(n_genes=len(GENES), embedding_dim=8, hidden_dim=8))
    index = torch.tensor([[0, 1], [2, 0]])
    states = torch.randn(len(GENES), 8)
    adjacency = sparse_adjacency(
        torch.tensor([[0, 1, 2], [1, 2, 3]]), torch.ones(3), len(GENES)
    )
    control = torch.zeros(2, len(GENES))
    expression, delta = model(control, index, adjacency, None, adjacency, gene_states=states)
    assert expression.shape == (2, len(GENES)) == delta.shape
    assert torch.allclose(expression, control + delta)


def test_the_three_arms_train_and_report_perturbation_level_metrics(tmp_path):
    """The comparison the design asks for: base, ungated three-graph, gated."""
    frame = synthetic_screen()
    task = build_task(frame, condition_column="condition", gene_columns=GENES)
    examples, report = build_examples(frame, task, context_columns=["cell_type"])
    labels = sorted({example.label for example in examples})
    splits_by_label = perturbation_splits(
        labels, mode="mixed", test_fraction=0.34, validation_fraction=0.17, seed=0
    )
    splits = split_examples(examples, splits_by_label)
    assert splits["train"] and splits["test"], "the synthetic screen must populate both"

    controls = frame[frame["condition"] == CONTROL][GENES].to_numpy(dtype=np.float32)
    graphs = {
        "G_C": coexpression_graph(controls, name="G_C", threshold=0.3, k=6),
        "G_S": coexpression_graph(controls * 0.9, name="G_S", threshold=0.3, k=6),
        "G_GO": go_graph(GENES, reference=small_go_graph(tmp_path), k=6),
    }
    config = PerturbationTrainingConfig(
        epochs=3,
        patience=3,
        batch_size=16,
        embedding_dim=8,
        hidden_dim=8,
        gnn_layers=1,
        top_k_deg=6,
        context_categorical={"cell_type": 2},
        seed=0,
    )
    results = run_perturbation_experiment(
        splits=splits, task=task, graphs=graphs, out_dir=tmp_path / "run", config=config,
        # the augmented arms need supplementary cells to generate labels for
        supplementary_controls=controls,
    )

    assert set(results["arms"]) == set(ARMS)
    for arm in ARMS:
        test = results["arms"][arm]["test"]
        assert test["n_perturbations"] >= 1
        for key in ("mse", "mse_deg", "pearson", "direction_accuracy", "top_k_overlap"):
            assert key in test["summary"]
    gated = results["arms"]["gears_augmented"]["history"]
    assert any("gate" in entry for entry in gated), "the gated arm must log its controller"
    # The mechanism is "teacher labels the supplementary cells", so the two
    # augmented arms must report a synthetic loss; the base arm must not.
    assert all("train_synthetic_loss" in entry for entry in gated)
    assert all(
        "train_synthetic_loss" in entry
        for entry in results["arms"]["gears_augmented_ungated"]["history"]
    )
    assert all(
        "train_synthetic_loss" not in entry
        for entry in results["arms"]["gears_base"]["history"]
    )
    assert "graph_edge_overlap" in results and results["graph_edge_overlap"]["shared"] >= 0
    written = json.loads((tmp_path / "run" / "perturbation_results.json").read_text())
    assert set(written["arms"]) == set(ARMS)
    contract = write_contract(
        tmp_path / "run", task=task, splits=splits_by_label, report=report
    )
    assert contract.exists()


def test_a_perturbation_question_is_routed_to_the_new_backend(tmp_path):
    """The agent's judgement: this is not a per-cell prediction task."""
    from kosmos.ppi.task_ontology import classify_task

    kind = classify_task(
        "Does knocking out CBL with CRISPR change the transcriptome, and can the "
        "response to a double knockout be predicted?"
    )
    assert kind.kind == "perturbation_response"
    assert kind.backend == "perturbation"
    assert kind.loss_mode == "graph_gated"

    # A plain per-cell prediction question still goes to the simple backend.
    simple = classify_task("Can the cell type be predicted from expression across donors?")
    assert simple.backend == "simple"


def test_the_backend_runs_from_tables_the_way_the_agent_will_call_it(tmp_path):
    frame = synthetic_screen()
    control = frame[frame["condition"] == CONTROL].copy()
    perturbed = frame[frame["condition"] != CONTROL].copy()
    # gold: controls + singles; supplementary: the combinations, treated as a
    # second source (its own control cells give the auxiliary graph)
    gold = pd.concat([control, perturbed[perturbed["condition"].isin(SINGLES)]])
    supp = pd.concat([control, perturbed[perturbed["condition"].str.contains("\\+")]])
    gold_path, supp_path = tmp_path / "gold.csv", tmp_path / "supp.csv"
    gold.to_csv(gold_path, index=False)
    supp.to_csv(supp_path, index=False)

    from kosmos.ppi.perturbation.run import run_perturbation_task

    results = run_perturbation_task(
        gold_path=gold_path,
        supplementary_paths=[supp_path],
        out_dir=tmp_path / "run",
        condition_column="condition",
        go_reference=small_go_graph(tmp_path),
        split_mode="mixed",
        test_fraction=0.34,
        validation_fraction=0.17,
        config=PerturbationTrainingConfig(
            epochs=3,
            patience=3,
            batch_size=16,
            embedding_dim=8,
            hidden_dim=8,
            gnn_layers=1,
            top_k_deg=6,
            seed=0,
        ),
    )

    assert set(results["arms"]) == set(ARMS)
    assert results["sources"]["gold"]["control_cells"] > 0
    assert "G_S" in results["graph_edge_overlap"] or results["graph_edge_overlap"]
    assert (tmp_path / "run" / "perturbation_contract.json").exists()
    assert (tmp_path / "run" / "graph_report.json").exists()


def test_a_supplementary_source_is_used_for_its_control_cells_not_its_labels(tmp_path):
    """The widening that motivates the auxiliary graph.

    A source in the same tissue with different perturbations -- or none at all
    -- is still useful, because the auxiliary graph comes from its control
    cells. What is *not* allowed is drawing the graph from perturbed cells: a
    screen with no recognisable control label gets its design regressed out
    instead, and if that is impossible the source is refused with a reason.
    """
    frame = synthetic_screen()
    from kosmos.ppi.perturbation.run import run_perturbation_task

    gold = frame[frame["condition"].isin([CONTROL, *SINGLES])]
    gold_path = tmp_path / "gold.csv"
    gold.to_csv(gold_path, index=False)

    # (a) an unperturbed atlas: no condition column at all
    atlas = frame[frame["condition"] == CONTROL].drop(columns=["condition"])
    atlas_path = tmp_path / "atlas.csv"
    atlas.to_csv(atlas_path, index=False)

    # (b) a scren with conditions but no control label: must be residualised
    other = frame[frame["condition"].isin(SINGLES)].copy()
    other["condition"] = ["cond_" + str(i % 3) for i in range(len(other))]
    other_path = tmp_path / "other_screen.csv"
    other.to_csv(other_path, index=False)

    results = run_perturbation_task(
        gold_path=gold_path,
        supplementary_paths=[atlas_path, other_path],
        out_dir=tmp_path / "run",
        condition_column="condition",
        go_reference=small_go_graph(tmp_path),
        test_fraction=0.34,
        validation_fraction=0.17,
        config=PerturbationTrainingConfig(
            epochs=2, patience=2, batch_size=16, embedding_dim=8, hidden_dim=8,
            gnn_layers=1, top_k_deg=6, seed=0,
        ),
    )

    sources = results["sources"]
    assert "unperturbed source" in sources["supp_0"]["how"]
    assert "residualized" in sources["supp_1"]["how"]
    assert sources["supp_0"]["edges"] > 0 and sources["supp_1"]["edges"] > 0
    assert results["graph_edge_overlap"]["shared"] >= 0


def test_the_retrieval_ask_for_a_perturbation_question_is_about_controls():
    from kosmos.ppi.perturbation.run import supplementary_requirement_text

    text = supplementary_requirement_text().lower()
    assert "control" in text and "tissue" in text
    assert "do not need to match" in text


def test_the_bridge_fetches_the_registered_screens_itself(tmp_path, monkeypatch):
    """A perturbation question with no plan stages gold and supp through the fetcher."""
    import kosmos.ppi.discovery_bridge as bridge
    from kosmos.ppi.perturbation import run as run_module
    from kosmos.ppi.perturbation import stage as stage_module

    frame = synthetic_screen()
    control = frame[frame["condition"] == CONTROL]
    gold = pd.concat([control, frame[frame["condition"].isin(SINGLES)]])
    supp = pd.concat([control, frame[frame["condition"].str.contains(r"\+")]])
    gold_path, supp_path = tmp_path / "gold.csv", tmp_path / "supp.csv"
    gold.to_csv(gold_path, index=False)
    supp.to_csv(supp_path, index=False)
    called = {}

    def fake_stage(*, gold, supplementary, condition_column="condition", echo=True):
        called["gold"] = gold.name
        called["supp"] = [source.name for source in supplementary]
        return gold_path, [supp_path], {"sources": {gold.name: {"url": gold.url}}}

    def fake_run(*, gold_path, supplementary_paths, out_dir, seed, **kwargs):
        return {
            "task": {"n_genes": len(GENES), "condition_column": "condition"},
            "split_sizes": {"train": 1, "validation": 1, "test": 1},
            "graph_edge_overlap": {"shared": 0},
            "arms": {
                "gears_base": {"test": {"summary": {"mse_deg": 0.30, "mse": 0.4, "pearson": 0.2,
                                                     "spearman": 0.1, "direction_accuracy": 0.8,
                                                     "top_k_overlap": 0.2}, "n_perturbations": 1},
                               "history": []},
                "gears_augmented_ungated": {"test": {"summary": {"mse_deg": 0.31, "mse": 0.4,
                                                                 "pearson": 0.2, "spearman": 0.1,
                                                                 "direction_accuracy": 0.8,
                                                                 "top_k_overlap": 0.2},
                                                     "n_perturbations": 1}, "history": []},
                "gears_augmented": {"test": {"summary": {"mse_deg": 0.29, "mse": 0.4, "pearson": 0.3,
                                                         "spearman": 0.2, "direction_accuracy": 0.9,
                                                         "top_k_overlap": 0.3},
                                             "n_perturbations": 1},
                                    "history": [{"epoch": 1, "gate": {"cosine": 0.4, "weight": 0.3,
                                                                      "gold_norm": 1.0, "increment_norm": 2.0,
                                                                      "active": 1.0},
                                                 "gate_active_fraction": 1.0}]},
            },
        }

    monkeypatch.setattr(stage_module, "stage_sources", fake_stage)
    # `run_perturbation_outcome` imports the runner inside the function, so the
    # patch has to land on the module the name is imported from.
    monkeypatch.setattr(run_module, "run_perturbation_task", fake_run)

    outcome = bridge.run_perturbation_outcome(
        plan_path=None, out_dir=tmp_path / "out", kind=_perturbation_kind(), seed=0, echo=False
    )

    assert outcome.ok is True
    assert called == {"gold": "norman", "supp": ["replogle_k562_essential"]}
    assert outcome.metrics["data_source"] == "registry"
    assert "gated_mse_deg" in outcome.metrics and outcome.metrics["value"] == 0.29


def test_run_data_task_routes_a_perturbation_question_to_the_registry(tmp_path, monkeypatch):
    """The path the CLI uses: a perturbation question with no plan.

    This is the seam that used to be dead -- `run_data_task` always resolved a
    plan before calling the perturbation backend, so the registered screens were
    never reached from the research loop. It now routes straight to the registry
    and leaves the column task's artifacts behind.
    """
    import kosmos.ppi.discovery_bridge as bridge
    from kosmos.ppi.perturbation import run as run_module
    from kosmos.ppi.perturbation import stage as stage_module

    frame = synthetic_screen()
    control = frame[frame["condition"] == CONTROL]
    gold = pd.concat([control, frame[frame["condition"].isin(SINGLES)]])
    supp = pd.concat([control, frame[frame["condition"].str.contains(r"\+")]])
    gold_path, supp_path = tmp_path / "gold.csv", tmp_path / "supp.csv"
    gold.to_csv(gold_path, index=False)
    supp.to_csv(supp_path, index=False)
    called = {}

    def fake_stage(*, gold, supplementary, condition_column="condition", echo=True):
        called["gold"] = gold.name
        return gold_path, [supp_path], {"sources": {gold.name: {"url": gold.url}}}

    def fake_run(*, gold_path, supplementary_paths, out_dir, seed, **kwargs):
        run_dir = Path(out_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.md").write_text("# summary\n", encoding="utf-8")
        figures = run_dir / "figures"
        figures.mkdir(exist_ok=True)
        figure = figures / "perturbation_overview.png"
        figure.write_bytes(b"png")

        def arm(mse_deg, pearson):
            return {
                "test": {
                    "summary": {
                        "mse": 0.4,
                        "mse_deg": mse_deg,
                        "pearson": pearson,
                        "spearman": 0.1,
                        "direction_accuracy": 0.8,
                        "top_k_overlap": 0.2,
                    },
                    "n_perturbations": 1,
                }
            }

        return {
            "task": {
                "n_genes": len(GENES),
                "condition_column": "condition",
                "control_labels": ["ctrl"],
            },
            "split_sizes": {"train": 3, "validation": 1, "test": 1},
            "graph_edge_overlap": {"shared": 2, "jaccard": 0.1},
            "sources": {"gold": {"path": str(gold_path), "cells": len(gold)}},
            "figures": [str(figure)],
            "arms": {
                "gears_base": arm(0.30, 0.2),
                "gears_augmented_ungated": arm(0.31, 0.2),
                "gears_augmented": arm(0.29, 0.3),
            },
        }

    monkeypatch.setattr(stage_module, "stage_sources", fake_stage)
    monkeypatch.setattr(run_module, "run_perturbation_task", fake_run)

    outcome = bridge.run_data_task(
        question="Does knocking out CBL with CRISPR change the transcriptome?",
        out_dir=tmp_path / "out",
        seed=0,
        echo=False,
    )

    assert outcome.ok, outcome.error
    assert outcome.kind.backend == "perturbation"
    assert called == {"gold": "norman"}
    assert outcome.metrics["data_source"] == "registry"
    # The unified artifacts, at the locations the column task uses.
    assert (tmp_path / "out" / "run" / "metrics.json").exists()
    assert (tmp_path / "out" / "run" / "summary.md").exists()
    assert (tmp_path / "out" / "data_report.md").exists()
    assert (tmp_path / "out" / "data_report.json").exists()
    assert outcome.summary_path.name == "summary.md"
    assert outcome.figures and outcome.figures[0].exists()
    returned = outcome.as_return_value()
    assert returned["data_source"] == "data_task"
    assert returned["staging"] == "registry"
    assert "gated_mse_deg" in returned



def _perturbation_kind():
    from kosmos.ppi.task_ontology import classify_task

    return classify_task("Does knocking out a gene change its module?")
