"""Split-half reliability of a gold screen's observed responses."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from kosmos.ppi.perturbation.dataprep.load import Sample
from kosmos.ppi.perturbation.dataprep.reliability import render, split_half


def _screen(seed=0):
    rng = np.random.default_rng(seed)
    genes = 300
    base = rng.gamma(2.0, 2.0, genes)
    strong = base.copy()
    strong[:30] *= 6  # a real response on 30 genes
    rows, labels = [], []
    for label, rate, n in (("NT", base, 400), ("STRONG", strong, 200), ("NULL", base, 200)):
        rows.append(rng.poisson(rate, size=(n, genes)))
        labels += [label] * n
    counts = sparse.csr_matrix(np.vstack(rows).astype(np.float32))
    return Sample(counts, [f"c{i}" for i in range(counts.shape[0])], [f"G{i}" for i in range(genes)], "fake"), labels


def test_a_real_response_is_reliable_and_noise_is_not():
    sample, labels = _screen()
    result = split_half(sample, labels, ["NT"], n_genes=300)
    per = result["per_perturbation"]
    assert per["STRONG"]["all"] > 0.8 and per["STRONG"]["de20"] > 0.5
    assert abs(per["NULL"]["all"]) < 0.3
    assert result["summary"]["perturbations"] == 2
    assert "top-20 DE" in render(result)


def test_small_perturbations_are_skipped():
    sample, labels = _screen()
    labels = ["NT" if l == "NT" else ("RARE" if i % 50 == 0 else l) for i, l in enumerate(labels)]
    result = split_half(sample, labels, ["NT"], min_cells=20, n_genes=300)
    assert "RARE" not in result["per_perturbation"]
