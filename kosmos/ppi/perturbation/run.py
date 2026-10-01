"""Run the perturbation backend from tables: the entry the agent calls.

Gold and supplementary tables come from the same plan the simple pipeline uses
(`plan.json`), so the data layer is shared. What changes is everything after
the plan: the contract is control/perturbation/Δ, the splits are by
perturbation, the graphs are built from control cells, and three arms are
trained and reported side by side.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..tabular import read_table
from .contract import (
    to_perturbed_gene,
    DEFAULT_CONTROL_LABELS,
    build_examples,
    build_task,
    is_control as is_control_value,
    parse_condition,
    perturbation_splits,
    split_examples,
    write_contract,
)
from .graphs import (
    GO_REFERENCE,
    coexpression_graph,
    edge_overlap,
    go_graph,
    identity_graph,
    one_hot,
    residualize,
    write_graph_report,
)
from .train import PerturbationTrainingConfig, run_perturbation_experiment

#: Columns that are never features and never perturbed genes.
CONTEXT_HINTS = ("cell_type", "celltype", "donor", "donor_id", "batch", "batch_id", "cell_id", "dose", "time", "timepoint")
#: The condition-column vocabulary lives in `modality.py` so the fetcher can use
#: it without importing torch.
from .modality import CONDITION_HINTS  # noqa: E402


def supplementary_requirement_text() -> str:
    """What a supplementary source has to provide for the augmented arm.

    This is the sentence the retrieval stage should carry for a perturbation
    question: the auxiliary graph is built from **control** cells, so a source
    is useful when it was measured in the same tissue/cell type and contains
    unperturbed cells. Its perturbations do not have to match the gold's, it
    does not need the gold's label column, and an untreated atlas is a perfectly
    good source.
    """
    return (
        "supplementary evidence for a co-expression graph: a dataset of the same "
        "tissue and cell type that contains control or untreated cells (an "
        "unperturbed atlas qualifies). Its perturbations do not need to match "
        "the gold's and it does not need the gold's label column; it only has to "
        "measure the same gene panel."
    )


#: A value that names a guide: letters and digits, e.g. `STAT2g1`, `NTg5`.
_GUIDE_VALUE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,19}$")


def _guide_like(value: str) -> bool:
    return bool(_GUIDE_VALUE.fullmatch(value)) and any(character.isdigit() for character in value)


def guess_condition_column(frame: pd.DataFrame, *, exclude: Sequence[str] = ()) -> str | None:
    """The column naming the perturbation: by its name, then by its values.

    A screen's condition column is usually named for what it holds (`gene`,
    `guide`, `condition`). When it is not -- ECCITE's guide columns are
    `GO_lenti_maxID` / `GO_cite_maxID`, and their labels are `STAT2g1`,
    `eGFPg1`, `NTg5` -- the values still say what the column is: a handful of
    strings, at least one of which means "nothing was perturbed" and the rest of
    which name a guide. Both pairs of conditions are required, so a hashing or
    donor column is not mistaken for a perturbation.
    """
    blocked = {str(name) for name in exclude}
    for name in CONDITION_HINTS:
        if name in frame.columns and str(name) not in blocked:
            return name
    best: tuple[int, str] | None = None
    for column in frame.columns:
        if str(column) in blocked:
            continue
        series = frame[column]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        values = [value for value in series.dropna().astype(str).unique() if value.strip()]
        if not 1 < len(values) <= 200:
            continue
        controls = sum(1 for value in values if is_control_value(value))
        if not controls:
            continue
        guides = sum(1 for value in values if _guide_like(value))
        if guides * 2 < len(values):
            continue
        score = 5 * controls + guides
        if best is None or score > best[0]:
            best = (score, str(column))
    return best[1] if best else None


def feature_columns(frame: pd.DataFrame, condition_column: str) -> list[str]:
    """The measured genes: numeric columns that are not context or the label.

    A joined screen records which columns came from the measurements
    (`assemble.join_screen`), and that is the reading that keeps the label
    table's own QC columns -- `nCount_RNA`, `percent.mito`, `S.Score` -- out of
    the gene panel. Only when nothing recorded them does this fall back to "every
    numeric column that is not the label".
    """
    curated = frame.attrs.get("expression_columns")
    if curated:
        present = {str(column) for column in frame.columns}
        listed = [str(column) for column in curated if str(column) in present]
        if listed:
            return listed
    numeric = frame.select_dtypes(include="number")
    blocked = {condition_column, *CONTEXT_HINTS}
    return [str(column) for column in numeric.columns if str(column) not in blocked]


def gene_coverage(
    frame: pd.DataFrame,
    column: str,
    genes: set[str],
    control_labels: Sequence[str],
) -> tuple[float, float, int]:
    """How much of a column's non-control values names a gene the panel measures.

    Two scores, because they mean different things: `mapped` counts a guide id
    as the gene it targets (`ATF2g1` -> `ATF2`), `direct` counts only values that
    are already measured genes. A screen that publishes both columns gets the
    one that needs no interpretation, at equal coverage.
    """
    values = [value for value in frame[column].dropna().astype(str).unique() if value.strip()]
    named = [value for value in values if not is_control_value(value, control_labels)]
    if not named:
        return 0.0, 0.0, 0
    hits = sum(1 for value in named if to_perturbed_gene(value, genes) in genes)
    direct = sum(1 for value in named if value in genes)
    return hits / len(named), direct / len(named), len(named)


def align_condition_column(
    frame: pd.DataFrame,
    genes: set[str],
    current: str | None,
    control_labels: Sequence[str],
) -> tuple[str | None, float]:
    """The column whose values are the genes the panel measures.

    A screen usually publishes both the guide that was delivered and the gene it
    targets (`guide_ID` = `ATF2g1`, `gene` = `ATF2`), and which one a reader
    calls "the perturbation" is a coin toss. The model can only encode a
    perturbation it can look up in the panel, so the column that names measured
    genes wins -- and a guide-id column still works, because `ATF2g1` is read as
    `ATF2` when the panel has `ATF2`.
    """
    best = str(current) if current else None
    best_score = (0.0, 0.0)
    if best and best in frame.columns:
        mapped, direct, _ = gene_coverage(frame, best, genes, control_labels)
        best_score = (mapped, direct)
    rows = max(1, len(frame))
    for column in frame.columns:
        if str(column) == best:
            continue
        series = frame[column]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        mapped, direct, named = gene_coverage(frame, str(column), genes, control_labels)
        # A perturbation column repeats: it says which perturbation each of many
        # cells carries. A column with one distinct value per row is a barcode,
        # and barcodes are not perturbations however well they match the panel's
        # cell names.
        if not named or named * 4 > rows:
            continue
        if (mapped, direct) > best_score:
            best, best_score = str(column), (mapped, direct)
    return best, best_score[0]


#: Library size every cell is scaled to before `log1p`, when the screen's
#: measurements are raw counts.
DEFAULT_TARGET_SUM = 1e4


def normalise_counts(
    frame: pd.DataFrame, genes: Sequence[str], *, target_sum: float | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Library-size normalise and `log1p` a raw-count screen.

    A screen's published matrix is usually raw UMI counts (the ECCITE
    macrophages run 5,000-30,000 counts per cell and are 88% zeros). The
    contract is defined on expression *differences*, and on raw counts those
    differences are dominated by how deeply each cell was sequenced, so the
    standard per-cell normalisation is applied and reported. A table that does
    not look like counts -- `verdict` says why -- is passed through untouched,
    because normalising an already-normalised matrix divides real units by a
    number that means nothing.
    """
    from ..singlecell import counts_verdict

    target = (
        float(target_sum)
        if target_sum is not None
        else float(os.getenv("KOSMOS_PERTURBATION_TARGET_SUM", DEFAULT_TARGET_SUM))
    )
    looks, why = counts_verdict(frame, [str(gene) for gene in genes])
    if not looks:
        return frame, {"applied": False, "why": why}
    values = frame[genes].to_numpy(dtype=np.float64, na_value=0.0)
    totals = values.sum(axis=1, keepdims=True)
    totals[totals <= 0] = 1.0
    out = frame.copy()
    out[genes] = pd.DataFrame(
        np.log1p(values / totals * target).astype(np.float32),
        index=frame.index,
        columns=list(genes),
    )
    return out, {
        "applied": True,
        "why": why,
        "target_sum": target,
        "how": f"library-size normalisation to {target:g} then log1p",
    }


#: How many genes the panel may hold. A published screen measures every gene the
#: assay saw -- the ECCITE-seq macrophages have 18,649 -- and both graphs are
#: quadratic in the panel (18,649 nodes is 348M candidate edges, minutes of
#: correlation for a matrix whose bottom half is all zeros) and the decoder is a
#: layer over every gene. The panel is therefore capped, by variance across the
#: *control* cells (the standard HVG reading, and the cells the graph itself is
#: built from). 0 disables the cap; `KOSMOS_PERTURBATION_MAX_GENES` overrides it.
DEFAULT_MAX_PANEL_GENES = 2000


def max_panel_genes() -> int:
    raw = os.getenv("KOSMOS_PERTURBATION_MAX_GENES", str(DEFAULT_MAX_PANEL_GENES))
    try:
        return max(0, int(float(raw)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_PANEL_GENES


def select_panel(
    frame: pd.DataFrame,
    genes: Sequence[str],
    *,
    control_mask: pd.Series | None = None,
    always_include: Sequence[str] = (),
    limit: int | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Cap the measured genes to the most variable ones across the controls.

    The genes a screen perturbed are always kept, however quiet they are: the
    model encodes a perturbation from the *panel* embeddings of the genes it
    names, so a perturbed gene outside the panel cannot be represented at all
    and its cells drop out of the training set.

    Which genes make the panel is a claim about what the model is asked to
    predict, so it is reported: `panel_selection` travels in the contract and in
    the data report.
    """
    measured = [str(gene) for gene in genes]
    present = set(measured)
    must = [gene for gene in dict.fromkeys(str(g) for g in always_include) if gene in present]
    cap = max_panel_genes() if limit is None else int(limit)
    info: dict[str, Any] = {
        "measured": len(measured),
        "selected": len(measured),
        "cap": cap,
        "perturbed_genes_kept": len(must),
        "how": "every measured gene",
    }
    if cap <= 0 or len(measured) <= cap:
        return measured, info
    keep = set(must)
    remaining = cap - len(keep)
    how = f"top {cap} by variance"
    if remaining > 0:
        values = frame[measured].to_numpy(dtype=np.float32)
        mask = None if control_mask is None else control_mask.to_numpy(dtype=bool)
        if mask is not None and mask.any():
            values = values[mask]
        variance = np.nan_to_num(np.nanvar(values, axis=0), nan=-np.inf)
        ranked = [
            measured[index]
            for index in np.argsort(-variance)
            if measured[index] not in keep
        ][:remaining]
        keep.update(ranked)
        how = (
            f"top {cap - len(must)} by variance across the "
            f"{'control ' if mask is not None else ''}cells plus the "
            f"{len(must)} perturbed gene(s) "
            f"(KOSMOS_PERTURBATION_MAX_GENES={cap})"
        )
    selected = [gene for gene in measured if gene in keep]
    info.update(
        {
            "selected": len(selected),
            "how": how,
            "controls_used": int(control_mask.sum()) if control_mask is not None else 0,
        }
    )
    return selected, info


def run_perturbation_task(
    *,
    gold_path: str | Path,
    out_dir: str | Path,
    supplementary_paths: Sequence[str | Path] = (),
    supplementary_frames: Sequence[pd.DataFrame] = (),
    condition_column: str | None = None,
    control_labels: Sequence[str] = DEFAULT_CONTROL_LABELS,
    context_columns: Sequence[str] = ("cell_type", "donor", "batch"),
    split_mode: str = "mixed",
    test_fraction: float = 0.2,
    validation_fraction: float = 0.1,
    min_cells_per_perturbation: int = 2,
    go_reference: str | Path = GO_REFERENCE,
    go_k: int = 20,
    coexpress_threshold: float = 0.4,
    coexpress_k: int = 20,
    seed: int = 42,
    config: PerturbationTrainingConfig | None = None,
    modality: dict[str, Any] | None = None,
    expression_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the contract and graphs from tables, then train the three arms.

    `expression_path` is for a screen published as two files: the label table
    (barcodes plus the perturbation each cell carried) and the measurements
    (barcodes by genes). They are joined on the barcode before anything else
    looks at the data, because neither half alone is a training table.

    `supplementary_frames` is the same pair already joined by the caller -- how a
    second screen in the same panel becomes the *unlabeled* source the augmented
    arms learn from, without writing a two-gigabyte copy to disk to hand it over
    as a path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    from kosmos.core.diagnostics import stage

    with stage(
        f"reading the screen's tables ({Path(str(gold_path)).name}"
        + (f" + {Path(str(expression_path)).name}" if expression_path else "")
        + ")"
    ):
        gold = read_table(gold_path)
        condition_column = condition_column or guess_condition_column(gold)
        if expression_path is not None and condition_column:
            from .assemble import join_screen

            label_table = gold
            expression_table = read_table(expression_path)
            gold = join_screen(
                label_table, expression_table, condition_column=condition_column
            )
            gold.attrs["assembled_from"] = [str(gold_path), str(expression_path)]
        if modality:
            modality = {**modality}
            modality["assembled_from"] = [
                str(Path(str(gold_path))),
                str(Path(str(expression_path))),
            ]
    if not condition_column or condition_column not in gold.columns:
        raise ValueError(
            "no perturbation column: expected one of "
            f"{list(CONDITION_HINTS)} in {Path(str(gold_path)).name}"
        )
    measured = feature_columns(gold, condition_column)
    if not measured:
        raise ValueError("no numeric gene columns to measure a perturbation against")
    # The label the plan carries is a guess about which column holds the
    # perturbation; the panel decides whether it is one the model can encode.
    aligned, coverage = align_condition_column(
        gold, set(measured), condition_column, control_labels
    )
    if aligned and aligned != str(condition_column):
        print(
            f"# perturbation: {aligned!r} names the genes this panel measures "
            f"({coverage:.0%} of its values), so it is the perturbation column, "
            f"not {str(condition_column)!r}",
            flush=True,
        )
        condition_column = aligned
    with stage(
        f"normalising the measurements and choosing the panel ({len(measured):,} measured gene(s))"
    ):
        gold, preprocessing = normalise_counts(gold, measured)
    control_mask = gold[condition_column].map(
        lambda value: is_control_value(value, control_labels)
    )
    # The gold has to carry *both* halves of a perturbation example: which
    # perturbation each cell carries, and the transcriptome it changed. A table
    # with the first and not the second (a cell-metadata file: barcodes plus QC
    # columns) passes every other check and cannot be trained on.
    perturbation_genes: set[str] = set()
    for value in gold[condition_column].astype(str).unique():
        if is_control_value(value, control_labels):
            continue
        for gene in parse_condition(value):
            perturbation_genes.add(to_perturbed_gene(gene, set(measured)))
    if perturbation_genes and not (perturbation_genes & set(measured)):
        raise ValueError(
            f"the gold names {len(perturbation_genes)} perturbation(s) "
            f"(e.g. {sorted(perturbation_genes)[:3]}) but none is among its "
            f"{len(measured)} measured column(s) (e.g. {list(measured)[:3]}): this "
            f"table has the labels and not the transcriptome -- its numeric "
            f"columns are metadata, and the expression matrix is in another file "
            f"(guide ids are read as the gene they target when the panel has it)"
        )
    genes, panel_selection = select_panel(
        gold, measured, control_mask=control_mask, always_include=sorted(perturbation_genes)
    )

    # The panel is shared by every dataset and every graph, so it is the
    # intersection of what the gold and the usable supplementary sources
    # measure. Two screens prepared by different pipelines often share only a
    # fraction of their symbols; training on the gold's own panel while the
    # auxiliary graph covers a fifth of it would compare an arm to itself.
    sources: dict[str, Any] = {
        "gold": {
            "path": str(gold_path),
            "cells": int(len(gold)),
        }
    }
    tables: list[tuple[int, Path, Any]] = []
    for index, path in enumerate(supplementary_paths):
        try:
            table = read_table(path)
        except Exception as error:  # noqa: BLE001 - unreadable source is reported
            sources[f"supp_{index}"] = {"path": str(path), "error": f"{type(error).__name__}: {error}"}
            continue
        tables.append((index, Path(path), table))
    for frame in supplementary_frames:
        index = len(tables)
        origin = str(frame.attrs.get("assembled_from_label") or "assembled screen")
        tables.append((index, Path(origin), frame))
        sources[f"supp_{index}"] = {
            "path": origin,
            "how": "joined from a label table and its own measurements",
            "assembled_from": frame.attrs.get("assembled_from") or [],
        }
    panel = list(genes)
    if tables:
        shared = set(genes)
        for _, _, table in tables:
            shared &= {str(column) for column in table.columns}
        panel = [gene for gene in genes if gene in shared]
        if len(panel) < 10:
            raise ValueError(
                f"the gold and the supplementary sources share only {len(panel)} "
                f"gene(s), which is not a panel: check that the sources use the "
                f"same gene symbols (an index of Ensembl ids next to symbols in "
                f"`condition` produces exactly this)."
            )
        # A narrower intersection is a fact about the sources, not a failure:
        # both arms train on this panel, and the share is recorded so a reader
        # can see how much of the gold the auxiliary graph actually covered.
    task = build_task(
        gold,
        condition_column=condition_column,
        gene_columns=panel,
        control_labels=control_labels,
    )
    examples, report = build_examples(
        gold,
        task,
        context_columns=context_columns,
        min_cells_per_perturbation=min_cells_per_perturbation,
    )
    # The panel is now narrower than the gold. That is a fact about the sources,
    # and it decides how much of the screen the auxiliary graph can even see, so
    # it is reported rather than left implicit.
    share = len(panel) / max(1, len(genes))
    report["preprocessing"] = preprocessing
    report["panel_in_gold"] = len(measured)
    report["panel_selected"] = len(genes)
    report["panel_selection"] = panel_selection
    report["panel_shared_with_supplementary"] = len(panel)
    report["panel_share_of_gold"] = round(share, 4)
    report["panel_per_source"] = {
        **{"gold": len(measured)},
        **{
            f"supp_{index}": int(
                sum(1 for gene in measured if gene in set(str(c) for c in table.columns))
            )
            for index, _, table in tables
        },
    }
    if share < 0.5 and tables:
        import logging

        message = (
            f"the shared panel is {len(panel)} of the gold's {len(genes)} gene(s) "
            f"({share:.0%}): the auxiliary graph can only reach that much of the "
            f"screen. Check that the sources use the same gene symbols."
        )
        report["panel_warning"] = message
        logging.getLogger(__name__).warning(message)
        # Printed: a panel that covers a fifth of the gold changes what the
        # numbers mean, and a reader watching the run has to see it.
        print(f"# WARNING: {message}", flush=True)
    if not examples:
        raise ValueError(
            "no perturbed cells with a control baseline; the gold needs control "
            "cells and perturbation identities"
        )

    # the gold graph must be built on the *panel*, not on the gold's own gene list:
    # the panel is what every source and every graph shares
    control_matrix = gold.loc[control_mask, panel].to_numpy(dtype=np.float32)
    with stage(f"building the co-expression graphs over {len(panel):,} panel gene(s)"):
        graphs = {
            "G_C": coexpression_graph(
                control_matrix, name="G_C", threshold=coexpress_threshold, k=coexpress_k
            )
        }
    # supplementary: prefer its control cells; otherwise compare the whole source
    sources["gold"].update(
        {
            "control_cells": int(control_mask.sum()),
            "genes": len(measured),
            "panel_after_intersection": len(panel),
        }
    )
    auxiliary: list[Any] = []
    # The first source that yields a graph also supplies the control cells the
    # augmented MLP baseline queries with teacher-generated synthetic labels.
    supplementary_controls: np.ndarray | None = None
    for index, path, table in tables:
        missing = [gene for gene in panel if gene not in table.columns]
        if missing:
            sources[f"supp_{index}"] = {
                "path": str(path),
                "error": f"missing {len(missing)} panel gene(s), e.g. {missing[:3]}",
            }
            continue
        supp_condition = condition_column if condition_column in table.columns else guess_condition_column(table)
        # What makes a supplementary source useful for the *co-expression* graph
        # is that it was measured in the same tissue and carries cells that were
        # not perturbed -- its own perturbation labels are irrelevant, which is
        # why this is a much wider net than "a table with the gold's columns".
        # Three cases, in order of preference:
        if supp_condition is None:
            # No perturbation column at all: an unperturbed atlas. Every cell is
            # a control by construction.
            rows = table[panel].to_numpy(dtype=np.float32)
            how = "all cells (no condition column: an unperturbed source)"
        else:
            is_control = table[supp_condition].map(
                lambda value: is_control_value(value, control_labels)
            )
            control_cells = int(is_control.sum())
            if control_cells >= 3:
                rows = table.loc[is_control, panel].to_numpy(dtype=np.float32)
                how = f"control cells ({control_cells})"
            else:
                # A screen without a recognisable control label. Using every cell
                # would draw the graph from the experimental design itself, so
                # the design is removed instead: perturbation identity, dose,
                # time and donor/batch are regressed out and the correlation is
                # taken from what is left.
                design_blocks = [
                    one_hot(table[supp_condition].astype(str)),
                ]
                for name in CONTEXT_HINTS:
                    if name in table.columns and name != supp_condition:
                        column = table[name]
                        design_blocks.append(
                            one_hot(column.astype(str))
                            if not pd.api.types.is_numeric_dtype(column)
                            else column.to_numpy(dtype=np.float64).reshape(-1, 1)
                        )
                design = np.hstack([block for block in design_blocks if block.size])
                try:
                    rows = residualize(table[panel].to_numpy(dtype=np.float32), design)
                    how = (
                        "residualized (no control label in this source: "
                        "perturbation/dose/time/donor/batch removed before correlating)"
                    )
                except ValueError as error:
                    sources[f"supp_{index}"] = {
                        "path": str(path),
                        "error": (
                            f"no usable control cells and residualisation failed "
                            f"({error}); this source needs controls or fewer "
                            f"nuisance columns to contribute a graph"
                        ),
                    }
                    continue
        name = f"G_S{index}" if index else "G_S"
        graph = coexpression_graph(
            rows, name=name, threshold=coexpress_threshold, k=coexpress_k
        )
        if supplementary_controls is None:
            supplementary_controls = np.asarray(rows, dtype=np.float32)
        auxiliary.append(graph)
        sources[f"supp_{index}"] = {
            "path": str(path),
            "cells": int(len(table)),
            "used_cells": int(rows.shape[0]),
            "how": how,
            "edges": graph.n_edges,
        }
    auxiliary_graphs: list = []
    if auxiliary:
        # One graph per source: incompatible sources are never pooled into one.
        # The trainer takes the first as the auxiliary branch; the rest travel in
        # the report, because pooling them would be a claim about compatibility.
        graphs["G_S"] = auxiliary[0]
        auxiliary_graphs = auxiliary[1:]
    else:
        # A screen with no second measurement: the augmented arms still run --
        # the trainer's six-arm contract is what the report and the metrics are
        # written against -- but there are no supplementary cells to label, so
        # they are the gold update on one more graph branch. Saying it here is
        # what keeps "the augmented arm did not help" from reading as a finding.
        graphs["G_S"] = graphs["G_C"]
        report["no_supplementary_source"] = (
            "no supplementary measurement shares the panel: the augmented arms "
            "have no unlabeled cells to learn from and reduce to the gold update"
        )
        print(
            "# WARNING: no supplementary source shares the panel; the augmented "
            "arms have nothing extra to learn from",
            flush=True,
        )
    go_path = Path(go_reference)
    with stage(f"building the GO similarity graph ({go_path.name})"):
        graphs["G_GO"] = (
            go_graph(panel, reference=go_path, k=go_k)
            if go_path.exists()
            else identity_graph(len(panel), name="G_GO")
        )

    labels = sorted({example.label for example in examples})
    splits_by_label = perturbation_splits(
        labels,
        mode=split_mode,
        test_fraction=test_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    splits = split_examples(examples, splits_by_label)
    if not splits["train"] or not splits["test"]:
        raise ValueError(
            "the splits left a side empty: "
            f"{ {name: len(values) for name, values in splits.items()} }; "
            "a perturbation screen needs several perturbations per side"
        )

    training = config or PerturbationTrainingConfig(seed=seed)
    # The MLP baselines are trained beside `arms`, not inside it: the message
    # counts what will actually be trained, or it under-reports every run that
    # has them on.
    arm_count = len(training.arms) + (
        3 if getattr(training, "include_mlp_baselines", False) else 0
    )
    with stage(
        f"training {arm_count} arms for up to {training.epochs} epoch(s)"
    ):
        results = run_perturbation_experiment(
            splits=splits,
            task=task,
            graphs=graphs,
            out_dir=out,
            config=training,
            supplementary_controls=supplementary_controls,
            modality=modality,
        )
    # The perturbation-type judgement belongs in the contract: it is a property
    # of the data (which screen, which assay) that the rest of the pipeline is
    # blind to, and the one thing a reader needs to know before trusting a
    # "predicted the knockout effect" claim.
    if modality:
        report["modality"] = modality
        provided = modality.get("provided") or {}
        if isinstance(provided, dict):
            gold_record = sources.setdefault("gold", {})
            gold_record["modality"] = provided.get("gold")
            if provided.get("gold_source"):
                gold_record["modality_source"] = provided.get("gold_source")
    write_contract(out, task=task, splits=splits_by_label, report=report, sources=sources)
    write_graph_report(
        out,
        {name: graph for name, graph in graphs.items() if hasattr(graph, "to_dict")},
        extra={
            "edge_overlap": (
                {"G_S": edge_overlap(graphs["G_C"], auxiliary[0])} if auxiliary else {}
            ),
            "additional_auxiliary_graphs": [graph.to_dict() for graph in auxiliary_graphs],
        },
    )
    results["sources"] = sources
    (out / "perturbation_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    return results
