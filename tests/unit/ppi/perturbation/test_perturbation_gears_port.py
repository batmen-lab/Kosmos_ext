"""The GEARS port: vocabulary, graphs, propagation, model, and the prepare stage."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")


def test_a_gears_condition_drops_the_ctrl_placeholder():
    from kosmos.ppi.perturbation.contract import parse_condition

    assert parse_condition("UBL5+ctrl") == ("UBL5",)
    assert parse_condition("CBL+CNN1") == ("CBL", "CNN1")
    assert parse_condition("ctrl") == ("ctrl",)


def test_the_perturbation_vocabulary_is_independent_of_the_panel():
    from kosmos.ppi.perturbation.contract import build_examples, build_task

    frame = pd.DataFrame(
        {"condition": ["ctrl", "ctrl", "X+ctrl", "X+ctrl", "Z+ctrl", "Z+ctrl"],
         "A": [1.0, 1.2, 0.2, 0.1, 1.0, 1.1], "B": [0.5, 0.4, 0.9, 1.0, 0.5, 0.6]}
    )
    # X is not measured, Z is outside the vocabulary
    task = build_task(frame, condition_column="condition", gene_columns=["A", "B"],
                      perturbation_vocabulary=["A", "X"])
    examples, report = build_examples(frame, task)
    assert {example.label for example in examples} == {"X"}
    # Z's cells are dropped, and the report says so (it was silent before)
    assert report["outside_vocabulary"] == ["Z"] and report["outside_vocabulary_cells"] == 2
    assert task.n_perturbation_genes == 2 and task.perturbation_to_index["X"] == 1


def test_propagation_matches_pyg_gcn_norm_source_to_target():
    from kosmos.ppi.perturbation.model import sparse_adjacency

    # edges 0->1 (w=2); node 1 already has a self loop (w=0.5); 0 and 2 get weight-1 loops
    edge_index = torch.tensor([[0, 1], [1, 1]])
    matrix = sparse_adjacency(edge_index, torch.tensor([2.0, 0.5]), 3).to_dense()
    degree = torch.tensor([1.0, 2.5, 1.0])  # weighted in-degree incl. loops
    assert matrix[1, 0] == pytest.approx(2.0 / (degree[0] * degree[1]).sqrt())
    assert matrix[1, 1] == pytest.approx(0.5 / degree[1])
    assert matrix[0, 1] == 0  # no message against the edge direction
    assert matrix[2, 2] == pytest.approx(1.0)


def test_graphs_keep_the_top_k_plus_one_sources_per_target_directed():
    from kosmos.ppi.perturbation.graphs import _gears_top_k_edges

    similarity = np.array([[1.0, 0.9, 0.5], [0.9, 1.0, 0.1], [0.5, 0.1, 1.0]])
    edge_index, weights = _gears_top_k_edges(similarity, k=1, threshold=0.4)
    edges = set(map(tuple, edge_index.T.tolist()))
    # target 2 keeps itself and source 0 (0.5 > 0.4); target 1 keeps itself and 0
    assert (0, 2) in edges and (2, 2) in edges and (1, 2) not in edges
    assert (0, 1) in edges and all(w > 0.4 for w in weights)


def test_the_model_encodes_unmeasured_perturbed_genes_and_pads_with_minus_one():
    from kosmos.ppi.perturbation.model import GearsModel, ModelConfig, sparse_adjacency

    n_genes, n_perts = 5, 8
    gene_adj = sparse_adjacency(torch.tensor([[0, 1], [1, 0]]), torch.ones(2), n_genes)
    go_adj = sparse_adjacency(torch.tensor([[5], [6]]), torch.ones(1), n_perts)
    model = GearsModel(ModelConfig(n_genes=n_genes, n_perturbations=n_perts, hidden_dim=8,
                                   condition_on_control=False)).eval()
    control = torch.zeros(3, n_genes)
    index = torch.tensor([[7, -1], [6, -1], [6, 7]])  # vocabulary ids beyond the panel
    _, delta = model(control, index, gene_adj, None, go_adj)
    assert delta.shape == (3, n_genes)
    assert not torch.allclose(delta[0], delta[1])


def test_the_prepare_stage_writes_tables_and_the_study(tmp_path, monkeypatch):
    anndata = pytest.importorskip("anndata")
    import scripts.prepare_perturbation as prepare

    rng = np.random.default_rng(0)
    def screen(path, genes, ids, conditions):
        adata = anndata.AnnData(
            X=rng.random((len(conditions), len(genes))).astype(np.float32),
            obs=pd.DataFrame({"condition": conditions, "cell_type": "K562"},
                             index=[f"c{i}" for i in range(len(conditions))]),
            var=pd.DataFrame({"gene_name": genes}, index=ids),
        )
        adata.write_h5ad(path)
    gold_path, supp_path = tmp_path / "gold.h5ad", tmp_path / "supp.h5ad"
    screen(gold_path, ["A", "B", "C"], ["e1", "e2", "e3"], ["ctrl"] * 3 + ["A+ctrl"] * 3)
    screen(supp_path, ["B", "D"], ["e2", "e9"], ["ctrl"] * 2 + ["B+ctrl"] * 2)
    stage1 = tmp_path / "stage1"
    stage1.mkdir()
    (stage1 / "plan.json").write_text(json.dumps({
        "task": {"objective": "q", "backend": "perturbation"},
        "gold": [{"path": str(gold_path), "dataset": "toy"}],
        "supplementary": [{"path": str(supp_path), "dataset": "toysupp"}],
    }))
    monkeypatch.setattr("kosmos.ppi.perturbation.graphs.GO_REFERENCE", tmp_path / "missing.csv")
    assert prepare.main(["--stage1", str(stage1), "--out", str(tmp_path / "out")]) == 0
    plan = json.loads((tmp_path / "out" / "plan.json").read_text())
    assert plan["study"]["split_mode"] == "unseen_single" and plan["study"]["test_fraction"] == 0.25
    gold = pd.read_parquet(plan["gold"][0]["path"])
    assert list(gold.columns) == ["condition", "cell_type", "A", "B", "C"]
    supp = pd.read_parquet(plan["supplementary"][0]["path"])
    assert list(supp.columns) == ["A", "B", "C"] and len(supp) == 4
    # A and C are not measured by the supplementary screen: filled with the gold control mean
    control_mean = gold[gold.condition == "ctrl"][["A", "C"]].mean().to_numpy()
    assert np.allclose(supp[["A", "C"]].to_numpy(), control_mean, atol=1e-6)
