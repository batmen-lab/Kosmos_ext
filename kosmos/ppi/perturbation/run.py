"""Run the perturbation backend from tables: the entry the agent calls.

Gold and supplementary tables come from the same plan the simple pipeline uses
(`plan.json`), so the data layer is shared. What changes is everything after
the plan: the contract is control/perturbation/Δ, the splits are by
perturbation, the graphs are built from control cells, and three arms are
trained and reported side by side.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..tabular import read_table
from .contract import (
    DEFAULT_CONTROL_LABELS,
    build_examples,
    build_task,
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
CONDITION_HINTS = ("condition", "perturbation", "guide", "target_gene", "gene_target", "sgRNA", "compound_1")


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


def guess_condition_column(frame: pd.DataFrame) -> str | None:
    for name in CONDITION_HINTS:
        if name in frame.columns:
            return name
    return None


def feature_columns(frame: pd.DataFrame, condition_column: str) -> list[str]:
    """The measured genes: numeric columns that are not context or the label."""
    numeric = frame.select_dtypes(include="number")
    blocked = {condition_column, *CONTEXT_HINTS}
    return [str(column) for column in numeric.columns if str(column) not in blocked]


def run_perturbation_task(
    *,
    gold_path: str | Path,
    out_dir: str | Path,
    supplementary_paths: Sequence[str | Path] = (),
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
) -> dict[str, Any]:
    """Build the contract and graphs from tables, then train the three arms."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gold = read_table(gold_path)
    condition_column = condition_column or guess_condition_column(gold)
    if not condition_column or condition_column not in gold.columns:
        raise ValueError(
            "no perturbation column: expected one of "
            f"{list(CONDITION_HINTS)} in {Path(str(gold_path)).name}"
        )
    genes = feature_columns(gold, condition_column)
    if not genes:
        raise ValueError("no numeric gene columns to measure a perturbation against")

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
    report["panel_in_gold"] = len(genes)
    report["panel_shared_with_supplementary"] = len(panel)
    report["panel_share_of_gold"] = round(share, 4)
    report["panel_per_source"] = {
        **{"gold": len(genes)},
        **{
            f"supp_{index}": int(
                sum(1 for gene in genes if gene in set(str(c) for c in table.columns))
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

    control_mask = gold[condition_column].map(lambda value: str(value).lower() in {c.lower() for c in control_labels})
    # the gold graph must be built on the *panel*, not on the gold's own gene list:
    # the panel is what every source and every graph shares
    control_matrix = gold.loc[control_mask, panel].to_numpy(dtype=np.float32)
    graphs = {
        "G_C": coexpression_graph(
            control_matrix, name="G_C", threshold=coexpress_threshold, k=coexpress_k
        )
    }
    # supplementary: prefer its control cells; otherwise compare the whole source
    sources["gold"].update(
        {
            "control_cells": int(control_mask.sum()),
            "genes": len(genes),
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
                lambda value: str(value).lower() in {c.lower() for c in control_labels}
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
    go_path = Path(go_reference)
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
