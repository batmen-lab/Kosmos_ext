"""The three graphs, and the statistics that make them reproducible.

  * **gold co-expression** (G_C) -- from the control cells of the gold training
    data only. Control cells are the unperturbed reference: a correlation
    measured across perturbed cells mixes the perturbation effect into the
    graph, which is the one thing the graph is meant to be independent of.
  * **auxiliary co-expression** (G_S) -- from supplementary cells. Control cells
    are preferred; if a source is mostly perturbed cells, the correlation is
    taken from residuals of a linear model that removes perturbation, dose, time
    and batch, so what is left is the cell-to-cell covariation rather than the
    treatment structure.
  * **GO annotation** (G_GO) -- the Jaccard similarity graph over Gene Ontology
    annotations that GEARS uses (`go_essential_all.csv`, `source,target,
    importance`), restricted to the panel and capped to the top-k edges per
    target exactly as GEARS does.

A graph is stored as the pair `(edge_index, edge_weight)` with a symmetric
normalisation applied at use time, together with the statistics that describe it
(threshold, k, edge count, density, degree quantiles). Every graph is written to
the run's artifacts, and the edge overlap between G_C and G_S is reported: two
graphs that share no edges cannot inform each other, and the overlap is the
first thing to check before blaming the model.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: GEARS uses |Pearson| >= 0.4 for the co-expression graph and the top-20
#: similar genes for the GO graph; the same defaults are kept here so a
#: comparison against its numbers is a comparison of data, not of thresholds.
DEFAULT_COEXPRESS_THRESHOLD = 0.4
DEFAULT_COEXPRESS_K = 20
DEFAULT_GO_K = 20

GO_REFERENCE = Path("data/perturbation/reference/go_essential_all.csv")


@dataclass
class GraphArtifact:
    """A graph plus what it was built from."""

    name: str
    edge_index: np.ndarray      # (2, E) int64
    edge_weight: np.ndarray     # (E,) float32
    stats: dict[str, Any]

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "n_edges": self.n_edges, **self.stats}

    def save(self, directory: str | Path) -> Path:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.name}.npz"
        np.savez_compressed(
            path, edge_index=self.edge_index, edge_weight=self.edge_weight
        )
        (out / f"{self.name}.json").write_text(
            json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        return path


def _edge_stats(
    edge_index: np.ndarray, edge_weight: np.ndarray, n_nodes: int, **extra: Any
) -> dict[str, Any]:
    if edge_index.size:
        degrees = np.bincount(edge_index[0], minlength=n_nodes)
    else:
        degrees = np.zeros(n_nodes, dtype=int)
    return {
        "n_nodes": int(n_nodes),
        "density": float(edge_index.shape[1] / max(1, n_nodes * (n_nodes - 1))),
        "weight_mean": float(edge_weight.mean()) if edge_weight.size else 0.0,
        "weight_min": float(edge_weight.min()) if edge_weight.size else 0.0,
        "weight_max": float(edge_weight.max()) if edge_weight.size else 0.0,
        "degree_mean": float(degrees.mean()),
        "degree_p50": float(np.quantile(degrees, 0.5)),
        "degree_p90": float(np.quantile(degrees, 0.9)),
        "degree_max": int(degrees.max()),
        "isolated_nodes": int((degrees == 0).sum()),
        **extra,
    }


def _top_k_edges(
    similarity: np.ndarray, *, k: int, threshold: float, self_loops: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Keep each node's k strongest partners above `threshold`, symmetrised."""
    n = similarity.shape[0]
    keep = np.zeros_like(similarity, dtype=bool)
    for row in range(n):
        row_values = similarity[row].copy()
        if not self_loops:
            row_values[row] = 0.0
        take = min(k, n - (0 if self_loops else 1))
        if take > 0:
            top = np.argpartition(-row_values, take - 1)[:take]
            keep[row, top] = True
    keep &= similarity >= threshold
    keep = keep | keep.T
    if self_loops:
        keep[np.arange(n), np.arange(n)] = True
    rows, columns = np.nonzero(keep)
    weights = similarity[rows, columns].astype(np.float32)
    return np.vstack([rows, columns]).astype(np.int64), weights


def coexpression_graph(
    control_matrix: np.ndarray,
    *,
    name: str = "G_C",
    threshold: float = DEFAULT_COEXPRESS_THRESHOLD,
    k: int = DEFAULT_COEXPRESS_K,
) -> GraphArtifact:
    """|Pearson| kNN graph over genes, from control cells."""
    matrix = np.asarray(control_matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 3:
        raise ValueError(
            f"a co-expression graph needs at least 3 control cells, got {matrix.shape}"
        )
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    scale = np.sqrt((centered**2).sum(axis=0, keepdims=True))
    scale[scale == 0] = 1.0
    normalised = centered / scale
    similarity = np.abs(normalised.T @ normalised)
    np.fill_diagonal(similarity, 1.0)
    similarity = np.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0)
    edge_index, edge_weight = _top_k_edges(similarity, k=k, threshold=threshold)
    return GraphArtifact(
        name=name,
        edge_index=edge_index,
        edge_weight=edge_weight,
        stats=_edge_stats(
            edge_index,
            edge_weight,
            similarity.shape[0],
            kind="coexpression",
            threshold=float(threshold),
            k=int(k),
            source_cells=int(matrix.shape[0]),
            absolute=True,
        ),
    )


def residualize(
    values: np.ndarray, design: np.ndarray, *, intercept: bool = True
) -> np.ndarray:
    """Remove the fitted effect of `design` columns from every gene.

    Used when a supplementary source is a mixture of conditions: the
    covariation we want is the one left after perturbation, dose, time and batch
    are explained away, so the auxiliary graph is not a picture of the
    experimental design.
    """
    X = np.asarray(design, dtype=np.float64)
    if intercept:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
    Y = np.asarray(values, dtype=np.float64)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("design and values must have one row per cell")
    if X.shape[1] >= X.shape[0]:
        raise ValueError("the design has as many columns as cells; nothing to fit")
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return (Y - X @ beta).astype(np.float32)


def one_hot(values: Sequence[Any]) -> np.ndarray:
    levels = sorted({str(value) for value in values})
    index = {level: position for position, level in enumerate(levels)}
    out = np.zeros((len(values), len(levels)), dtype=np.float64)
    for row, value in enumerate(values):
        out[row, index[str(value)]] = 1.0
    return out


def go_graph(
    genes: Sequence[str],
    *,
    reference: str | Path = GO_REFERENCE,
    name: str = "G_GO",
    k: int = DEFAULT_GO_K,
    chunksize: int = 2_000_000,
) -> GraphArtifact:
    """The GEARS GO similarity graph, restricted to the panel.

    Reads the 12M-row `source,target,importance` file in chunks and keeps the
    top-k partners per target, exactly as GEARS' `get_similarity_network` does
    for `network_type='go'`.
    """
    import pandas as pd

    path = Path(reference)
    if not path.exists():
        raise FileNotFoundError(
            f"the GO graph is not staged at {path}. It is the file GEARS uses "
            f"(Harvard Dataverse 6934319, `go_essential_all.csv`); download it or "
            f"pass `--go-graph <path>`, or run without the GO branch and say so."
        )
    panel = {str(gene) for gene in genes}
    keep_rows: list[np.ndarray] = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        mask = chunk["source"].isin(panel) & chunk["target"].isin(panel)
        if mask.any():
            keep_rows.append(chunk.loc[mask, ["source", "target", "importance"]])
    if not keep_rows:
        raise ValueError("the GO graph has no edge between any two panel genes")
    frame = pd.concat(keep_rows, ignore_index=True)
    frame = frame.sort_values("importance", ascending=False).drop_duplicates(
        ["source", "target"]
    )
    top = frame.groupby("target", group_keys=False).head(k + 1)
    index = {str(gene): position for position, gene in enumerate(genes)}
    rows = top["source"].map(index).to_numpy()
    columns = top["target"].map(index).to_numpy()
    weights = top["importance"].to_numpy(dtype=np.float32)
    keep = ~(np.isnan(rows) | np.isnan(columns))
    rows, columns, weights = rows[keep].astype(np.int64), columns[keep].astype(np.int64), weights[keep]
    edge_index = np.vstack([rows, columns])
    edge_index = np.hstack([edge_index, edge_index[::-1]])
    edge_weight = np.concatenate([weights, weights])
    return GraphArtifact(
        name=name,
        edge_index=edge_index,
        edge_weight=edge_weight,
        stats=_edge_stats(
            edge_index,
            edge_weight,
            len(genes),
            kind="go_similarity",
            k=int(k),
            reference=str(path),
            reference_sha256=file_sha256(path)[:16],
            rows_read=int(len(frame)),
        ),
    )


def identity_graph(n_nodes: int, *, name: str = "G_I") -> GraphArtifact:
    """Self-loops only: what "no auxiliary graph" looks like as a graph."""
    index = np.arange(n_nodes, dtype=np.int64)
    return GraphArtifact(
        name=name,
        edge_index=np.vstack([index, index]),
        edge_weight=np.ones(n_nodes, dtype=np.float32),
        stats=_edge_stats(
            np.vstack([index, index]), np.ones(n_nodes, dtype=np.float32), n_nodes, kind="identity"
        ),
    )


def edge_overlap(first: GraphArtifact, second: GraphArtifact) -> dict[str, float]:
    """How much two graphs have in common -- undirected, ignoring weights."""
    def keys(graph: GraphArtifact) -> set[tuple[int, int]]:
        return {
            (min(int(a), int(b)), max(int(a), int(b)))
            for a, b in zip(graph.edge_index[0], graph.edge_index[1], strict=False)
        }

    left, right = keys(first), keys(second)
    union = left | right
    if not union:
        return {"edges_first": 0, "edges_second": 0, "shared": 0, "jaccard": 0.0, "of_first": 0.0}
    shared = len(left & right)
    return {
        "edges_first": len(left),
        "edges_second": len(right),
        "shared": shared,
        "jaccard": shared / len(union),
        "of_first": shared / max(1, len(left)),
        "of_second": shared / max(1, len(right)),
    }


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def write_graph_report(
    out_dir: str | Path, graphs: dict[str, GraphArtifact], *, extra: dict[str, Any] | None = None
) -> Path:
    """Every graph's statistics and provenance, beside the graph files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {name: graph.to_dict() for name, graph in graphs.items()}
    if extra:
        payload.update(extra)
    path = out / "graph_report.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
