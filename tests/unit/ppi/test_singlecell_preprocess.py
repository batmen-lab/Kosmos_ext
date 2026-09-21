"""Single-cell preprocessing, per source: what it does and how it is proved."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kosmos.ppi.singlecell import (
    SingleCellConfig,
    align_sources,
    looks_like_counts,
    prepare_source,
    write_figures,
)


def counts_frame(cells: int = 60, genes: int = 40, seed: int = 0, offset: int = 0):
    rng = np.random.default_rng(seed)
    data = rng.poisson(3.0, size=(cells, genes)).astype(np.int64)
    data[:5, :2] += 50  # a couple of highly variable genes
    frame = pd.DataFrame(data, columns=[f"G{i}" for i in range(genes)])
    frame["cell_id"] = [f"c{i}" for i in range(cells)]
    frame["label"] = [f"t{(i + offset) % 3}" for i in range(cells)]
    return frame


def test_counts_are_recognised_and_a_survey_table_is_not():
    frame = counts_frame(genes=600)
    features = [c for c in frame.columns if c.startswith("G")]
    assert looks_like_counts(frame, features)
    # Few features, or negative values: not a count matrix.
    assert not looks_like_counts(frame, features[:10])
    assert not looks_like_counts(frame.assign(G0=-1.0), features)


def test_a_barcode_column_does_not_confuse_the_check_or_the_recipe():
    """A per-cell table carries text beside the counts, and that is not a gene.

    Asking pandas for floats of `GATGACGGAC-GGTGGGAT` raised before anything was
    judged -- the run died at the door with "could not convert string to float".
    """
    frame = counts_frame(genes=600)
    frame["barcode"] = [f"GATGACGGAC-GGTGGGA{i:02d}" for i in range(len(frame))]
    names = [c for c in frame.columns if c.startswith("G")] + ["barcode"]

    assert looks_like_counts(frame, names)

    source = prepare_source(
        frame, names, name="with-barcode", config=SingleCellConfig(n_top_genes=20)
    )

    assert "barcode" not in source.all_genes
    assert source.stats.genes_non_numeric == 1
    assert any("hold text" in note for note in source.notes)


def test_the_recipe_runs_in_order_and_clips_negatives():
    frame = counts_frame()
    features = [c for c in frame.columns if c.startswith("G")]

    source = prepare_source(frame, features, name="donor-1", config=SingleCellConfig(n_top_genes=10))

    assert source.n_genes == 10
    assert source.matrix.shape == (len(frame), 10)
    assert source.matrix.min() >= 0.0  # negatives are gone
    assert source.matrix.dtype == np.float32
    assert any("z-score" in note for note in source.notes)
    # The numbers are the recipe, checked by hand for one gene: library-size
    # normalisation, log1p, a per-gene z-score, then negatives clipped to 0.
    gene = source.genes[0]
    raw = frame[gene].to_numpy(dtype=np.float64)
    totals = frame[list(features)].to_numpy(dtype=np.float64).sum(axis=1)
    totals[totals == 0] = 1.0
    logged = np.log1p(raw / totals * 1e4)
    expected = np.clip((logged - logged.mean()) / logged.std(), 0.0, None)
    assert np.allclose(source.matrix[:, 0], expected, atol=1e-6)


def test_standardization_is_per_source_not_pooled():
    """A donor's scaling must not be another donor's."""
    first = counts_frame(seed=1)
    second = counts_frame(seed=2)
    second[["G0", "G1", "G2", "G3"]] *= 20  # a deeper-sequenced donor
    features = [c for c in first.columns if c.startswith("G")]
    config = SingleCellConfig(n_top_genes=15)

    a = prepare_source(first, features, name="a", config=config)
    b = prepare_source(second, features, name="b", config=config)
    assert a.genes != b.genes  # each source selected on its own variance

    panel, (aligned_a, aligned_b) = align_sources([a, b])

    # The raw counts of one source are 20x the other's, and after the per-source
    # normalisation both are on the same scale -- which is the whole point of
    # doing it source by source.
    index = 0
    raw_a = float(aligned_a.counts[:, index].mean())
    raw_b = float(aligned_b.counts[:, index].mean())
    assert raw_b > raw_a * 3
    processed_a = float(aligned_a.matrix[:, index].mean())
    processed_b = float(aligned_b.matrix[:, index].mean())
    assert 0 < processed_a < 2 and 0 < processed_b < 2


def test_the_gaussian_step_is_per_gene_within_a_source():
    from kosmos.ppi.singlecell import _zscore_genes

    values = np.asarray([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    scaled = _zscore_genes(values)

    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(scaled.std(axis=0), 1.0, atol=1e-12)


def test_sources_are_cut_to_the_genes_they_all_selected():
    first = prepare_source(
        counts_frame(seed=1), [f"G{i}" for i in range(40)], name="a",
        config=SingleCellConfig(n_top_genes=12),
    )
    second = prepare_source(
        counts_frame(seed=2), [f"G{i}" for i in range(40)], name="b",
        config=SingleCellConfig(n_top_genes=12),
    )

    panel, aligned = align_sources([first, second], max_genes=8)

    assert 0 < len(panel) <= 8
    assert all(source.genes == panel for source in aligned)
    assert all(source.matrix.shape[1] == len(panel) for source in aligned)


def test_the_panel_is_ranked_across_sources_not_just_intersected():
    """Four noisy top-2000 lists intersect to ~700 genes; ranking recovers more.

    A gene that three donors selected and one missed is shared measurement --
    every source measured it -- so it belongs in the panel ahead of a gene only
    one donor's variance liked.
    """
    frames = [counts_frame(seed=seed) for seed in (1, 2, 3, 4)]
    features = [f"G{i}" for i in range(40)]
    prepared = [
        prepare_source(frame, features, name=f"d{i}", config=SingleCellConfig(n_top_genes=10))
        for i, frame in enumerate(frames)
    ]
    plain_intersection = set(prepared[0].genes)
    for source in prepared[1:]:
        plain_intersection &= set(source.genes)

    panel, aligned = align_sources(prepared, max_genes=12)

    # The panel is at least what a plain intersection gives, and never more than
    # the cap.
    assert len(panel) <= 12
    assert len(panel) >= min(12, len(plain_intersection))
    # Every gene in it was measured by every source.
    for source in aligned:
        assert set(source.genes) == set(panel)


def test_every_gene_is_kept_when_the_panel_is_smaller_than_the_cap():
    frame = counts_frame(genes=20)
    source = prepare_source(
        frame, [f"G{i}" for i in range(20)], name="tiny",
        config=SingleCellConfig(n_top_genes=2000),
    )
    assert source.n_genes == 20
    assert any("below the cap" in note for note in source.notes)


def test_the_figures_are_written(tmp_path):
    pytest.importorskip("matplotlib")
    first = prepare_source(
        counts_frame(seed=1), [f"G{i}" for i in range(40)], name="gold",
        config=SingleCellConfig(n_top_genes=15),
    )
    second = prepare_source(
        counts_frame(seed=2), [f"G{i}" for i in range(40)], name="donor-2",
        config=SingleCellConfig(n_top_genes=15),
    )
    panel, aligned = align_sources([first, second])
    labels = {"gold": np.asarray(counts_frame(seed=1)["label"])}

    written = write_figures(aligned, tmp_path / "figures", labels=labels)

    names = {path.name for path in written}
    assert names == {
        "fig01_cells_per_source.png",
        "fig02_gene_selection.png",
        "fig03_pca_sources.png",
        "fig04_gene_transform.png",
        "fig05_gene_overlap.png",
        "fig06_recipe_effect.png",
    }
    assert all(path.exists() and path.stat().st_size > 1000 for path in written)
    assert len(panel) > 0


def test_the_statistics_say_what_each_source_is(tmp_path):
    """The numbers a reader needs to judge the preprocessing."""
    frame = counts_frame(cells=80, genes=50, seed=3)
    features = [f"G{i}" for i in range(50)]
    source = prepare_source(
        frame, features, name="donor-1", config=SingleCellConfig(n_top_genes=12, min_cells=5)
    )

    stats = source.stats
    assert stats.cells == 80
    assert stats.genes_in_file == 50
    assert stats.genes_selected == 12
    assert 0 < stats.sparsity_raw < 1
    assert stats.median_counts_per_cell > 0
    # Every gene in this fixture is seen in many cells, so none is dropped, and
    # the line a person reads says so.
    assert stats.genes_dropped_rare == 0
    line = stats.describe()
    assert "cells" in line and "zeros" in line


def test_a_gene_seen_in_too_few_cells_is_dropped_and_counted(tmp_path):
    frame = counts_frame(cells=60, genes=20, seed=4)
    features = [f"G{i}" for i in range(20)]
    frame["G0"] = 0
    frame.loc[0, "G0"] = 7  # detected in one cell only

    source = prepare_source(
        frame, features, name="sparse", config=SingleCellConfig(n_top_genes=100, min_cells=3)
    )

    assert source.stats.genes_dropped_rare == 1
    assert "G0" not in source.all_genes


def test_the_report_writes_markdown_and_json(tmp_path):
    pytest.importorskip("matplotlib")
    from kosmos.ppi.singlecell import write_report

    frames = [counts_frame(seed=seed, offset=seed) for seed in (1, 2)]
    prepared = [
        prepare_source(frame, [f"G{i}" for i in range(40)], name=f"donor-{n}",
                       config=SingleCellConfig(n_top_genes=15))
        for n, frame in enumerate(frames)
    ]
    panel, aligned = align_sources(prepared)
    figures = write_figures(aligned, tmp_path / "figures")

    markdown, record = write_report(
        aligned, panel, tmp_path, figures=figures,
        config=SingleCellConfig(n_top_genes=15),
    )

    text = markdown.read_text()
    payload = __import__("json").loads(record.read_text())
    # The recipe, in order, and one line per dataset.
    assert "## Recipe" in text and "log1p" in text and "clip negative values to 0" in text
    assert "## Datasets" in text and "donor-0" in text and "donor-1" in text
    assert "Trained panel:" in text
    assert "fig05_gene_overlap.png" in text
    assert payload["panel_size"] == len(panel)
    assert [row["name"] for row in payload["sources"]] == ["donor-0", "donor-1"]
    assert set(payload["pairwise_selection_overlap"]) == {"donor-0", "donor-1"}
