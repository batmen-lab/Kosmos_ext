"""Run a prediction task on tables: labeled gold, plus unlabeled evidence.

The whole flow, with nothing about any particular dataset in it:

    labeled table(s)      --random split-->  train / validation / test
    unlabeled table(s)    -------------->   PPI correction term only

Two roles, decided by one fact -- whether a table carries the task's label
column:

  * **gold**: labeled. Trains the model, selects the epoch, and (through the
    random split above) supplies both the validation set and the final test set.
  * **supplementary**: unlabeled. Never enters the supervised loss. Its rows get
    pseudo-labels from a gold-only model and enter through the signed PPI
    correction, whose second term subtracts the bias those pseudo-labels carry.

With no supplementary table the same entry point runs plain supervised
training: the engine already fits a matched gold-only baseline on every run, so
the strategy is a description of the data rather than a second code path.

Distribution is not a split axis here. Where the supplementary rows came from is
provenance -- one dataset id per table, carried into the artifacts -- so a reader
can check afterwards which data a correction consumed without the training code
having to model it.
"""

import json
import logging
import time
from collections.abc import Callable, Iterable, Sequence
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..core.diagnostics import stage
from .features import FeatureEncoder
from .report import write_markdown
from .schemas import ExternalEvidenceDataset, GoldDataset, PPITrainingConfig
from .split import split_gold
from .tabular import header_columns
from .tabular import read_table as read_delimited
from .task_spec import TaskSpec
from .trainer import evaluate_final_test, run_ppi_experiment

logger = logging.getLogger(__name__)


def read_table(path: str | Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Load a delimited table, sniffing its separator (fetched data is not ours)."""
    return read_delimited(path, columns=columns)


def row_fingerprint(matrix, limit: int = 20000) -> str:
    """A digest of the rows themselves, independent of their order.

    Two files can hold the same observations and still differ byte for byte --
    a re-export, a different separator, a copy republished elsewhere. Such a
    table is not external evidence: its rows are already in the labeled set, so
    the correction term cancels against itself. Comparing encoded rows catches
    it without caring how the file was written.
    """
    sample = matrix[:limit]
    rows = sorted(tuple(float(value) for value in row) for row in sample)
    return sha256(repr(rows).encode()).hexdigest()


def load_gold(
    path: str | Path,
    task: TaskSpec,
    dataset_id: str | None = None,
    role: str = "gold_train",
    encoder: FeatureEncoder | None = None,
) -> GoldDataset:
    """Read a labeled table into an unsplit dataset of the given role.

    Pass an `encoder` to encode this table the way another table was encoded --
    that is what makes a pooled or held-out labeled table line up with the one
    the vocabulary was fitted on. Without one, the encoder is fitted here, on
    this table alone.
    """
    frame = read_table(path)
    return gold_from_frame(
        frame, task, dataset_id or Path(path).stem, role, encoder=encoder
    )


def gold_from_frame(
    frame: pd.DataFrame,
    task: TaskSpec,
    dataset_id: str,
    role: str = "gold_train",
    encoder: FeatureEncoder | None = None,
) -> GoldDataset:
    if not task.has_target(frame.columns):
        raise ValueError(
            f"{dataset_id} has no target column {task.target_column!r}, so it is "
            f"not labeled data for this task. Pass it as supplementary evidence "
            f"instead."
        )
    features = task.feature_names(frame.columns)
    if not features:
        raise ValueError(
            f"{dataset_id} has no feature columns for this task: every column is "
            f"the target, an identifier, or excluded"
        )
    encoder = encoder or FeatureEncoder.fit(frame, features)
    # pandas stores text columns as `object`, which GoldDataset rejects: labels
    # must be finite numbers or strings, and "object" is neither. Numeric label
    # columns keep their dtype (a regression target is a real number).
    labels = frame[task.target_column]
    # `Series.astype(str).to_numpy()` still yields dtype=object, which
    # GoldDataset rejects -- it wants finite numbers or real strings. So a text
    # label column is converted with numpy, and a numeric one keeps its dtype
    # (a regression target is a number, not a string).
    if pd.api.types.is_numeric_dtype(labels) and not pd.api.types.is_bool_dtype(labels):
        y = np.asarray(labels)
    else:
        y = np.asarray(labels, dtype=str)
    return GoldDataset(
        X=encoder.transform(frame),
        y=y,
        sample_ids=task.sample_ids(frame, dataset_id),
        role=role,
        dataset_id=dataset_id,
        feature_names=list(encoder.feature_names),
        groups=None,
        metadata={
            "task": task.description or str(task.target_column),
            "raw_features": list(encoder.raw_features),
            "categorical_features": encoder.categorical_features,
        },
    )


def load_supplementary(
    path: str | Path,
    task: TaskSpec,
    dataset_id: str | None = None,
    *,
    max_rows: int | None = None,
    reference_features: Sequence[str] | None = None,
    encoder: FeatureEncoder | None = None,
    rename: dict[str, str] | None = None,
) -> ExternalEvidenceDataset:
    """Read an unlabeled table into PPI evidence.

    A table that happens to carry the target column is still accepted: its
    labels are ignored, because this task's labels come from gold data only.
    The feature schema has to match the gold table exactly -- PPI mixes both in
    one model, so a column present in one and absent from the other is not a
    formatting detail. `encoder` is the gold's: it is what makes `thal` here
    mean the same column as `thal` there, and it is what puts an unseen level in
    the unknown bucket instead of off the end of the matrix.
    """
    # Only the columns the encoder was fitted on are parsed. On a single-cell
    # table that is the whole run: 12,062 of 129,923 columns reads in under two
    # seconds, where the file whole took minutes -- the step that looked like
    # "stuck in Executing experiments".
    wanted = list(reference_features) if reference_features else None
    if wanted and rename:
        # `rename` maps this table's spelling to the gold's, so the table's own
        # names are what has to be asked for.
        wanted = [*wanted, *(old for old, new in rename.items() if new in set(wanted))]
    if wanted and task.sample_id_column:
        # The row identifiers are not features, but the mirror check and the
        # provenance both need them: reading only the features and then asking
        # for the id column is how this broke the first time.
        wanted.append(task.sample_id_column)
    if wanted:
        # Never ask a parser for a column this file does not have: the schema
        # check below says which features are missing, in the language of the
        # task, and a parser error would replace that with "Usecols do not
        # match".
        present = header_columns(path)
        if present is not None:
            # `set(present)` outside the loop: rebuilding a 129,923-element set
            # once per requested column is 1.5 billion string hashes, which is
            # minutes of work that looks exactly like a hang.
            present_set = set(present)
            wanted = [name for name in wanted if name in present_set]
    frame = read_table(path, columns=wanted)
    if rename:
        # The plan matched this table's columns to the gold's by name; a
        # different export spelled the same measurement differently
        # (`concave points_mean` vs `concave_points_mean`).
        frame = frame.rename(columns=dict(rename))
    if max_rows:
        frame = frame.iloc[:max_rows]
    return supplementary_from_frame(
        frame,
        task,
        dataset_id or Path(path).stem,
        source=str(path),
        reference_features=reference_features,
        encoder=encoder,
    )


def supplementary_from_frame(
    frame: pd.DataFrame,
    task: TaskSpec,
    dataset_id: str,
    *,
    source: str | None = None,
    reference_features: Sequence[str] | None = None,
    encoder: FeatureEncoder | None = None,
) -> ExternalEvidenceDataset:
    if reference_features is not None:
        present_columns = set(frame.columns)
        missing = [c for c in reference_features if c not in present_columns]
        if missing:
            raise ValueError(
                f"{dataset_id} is missing {len(missing)} feature column(s) the "
                f"gold table has, e.g. {missing[:3]}"
            )
        features = list(reference_features)
    else:
        features = task.feature_names(frame.columns)
    encoder = encoder or FeatureEncoder.fit(frame, features)
    return ExternalEvidenceDataset(
        X=encoder.transform(frame),
        sample_ids=task.sample_ids(frame, dataset_id),
        evidence_route="DIRECT_PPI",
        source_dataset_id=dataset_id,
        feature_names=list(encoder.feature_names),
        groups=None,
        audit_metadata={
            "source": source or dataset_id,
            "had_target_column": task.has_target(frame.columns),
            "rows": int(len(frame)),
            "raw_features": list(encoder.raw_features),
            "categorical_features": encoder.categorical_features,
        },
    )


def default_pseudo_labeler(seed: int = 42, task_type: str = "classification"):
    """The gold-only model that labels the supplementary rows.

    Standardisation lives inside the pipeline so the scaler is fit on gold
    only -- which is also what keeps it out of the supplementary data's
    distribution.

    A regressor predicts the value; a classifier predicts the class. The
    teacher has to be the same kind of model as the task, or the pseudo-labels
    it produces are not values the loss can compare anything against.
    """
    if task_type == "regression":
        from sklearn.linear_model import Ridge

        return make_pipeline(StandardScaler(), Ridge(random_state=seed))
    return make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)
    )


def pool_labeled_with_encoder(
    paths: Sequence[str | Path], task: TaskSpec, dataset_id: str = "pooled"
) -> tuple[GoldDataset, FeatureEncoder]:
    """Pool several labeled tables into one, when they share a feature schema.

    Using more data is the point, so tables that describe the same features are
    concatenated rather than one being picked. Tables that do not match are
    refused with the difference named: silently dropping one would quietly
    shrink the training set, and silently merging mismatched ones would train on
    misaligned columns.

    The comparison is on the *raw* columns, and the encoder is fitted once on the
    pooled rows, so every table in the pool is encoded the same way. Fitting an
    encoder per table would give each its own levels and its own width, which is
    the misalignment this function exists to prevent.

    Returns the pooled dataset and the encoder it was built with: the caller
    needs the second one to encode supplementary tables the same way.
    """
    frames = [read_table(path) for path in paths]
    for frame, path in zip(frames, list(paths), strict=False):
        if not task.has_target(frame.columns):
            raise ValueError(
                f"{Path(path).stem} has no target column "
                f"{task.target_column!r}, so it is not labeled data for this "
                f"task. Pass it as supplementary evidence instead."
            )
    features = task.feature_names(frames[0].columns)
    for frame, path in zip(frames[1:], list(paths)[1:], strict=False):
        own = task.feature_names(frame.columns)
        if own != features:
            own_set, wanted_set = set(own), set(features)
            missing = [c for c in features if c not in own_set]
            extra = [c for c in own if c not in wanted_set]
            raise ValueError(
                f"{path} does not share the pooled feature schema "
                f"(missing={missing[:3]}, extra={extra[:3]}); pool only tables "
                f"with identical features"
            )
    encoder = FeatureEncoder.fit(
        pd.concat(frames, ignore_index=True), features
    )
    datasets = [
        gold_from_frame(frame, task, Path(path).stem, encoder=encoder)
        for frame, path in zip(frames, list(paths), strict=False)
    ]
    labels = {dataset.y.dtype.kind for dataset in datasets}
    y = (
        np.concatenate([dataset.y for dataset in datasets])
        if len(labels) == 1
        else np.concatenate([np.asarray(dataset.y, dtype=str) for dataset in datasets])
    )
    sample_ids = np.concatenate([dataset.sample_ids for dataset in datasets])
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(
            "pooled tables share sample identifiers, so rows cannot be told "
            "apart; set task.sample_id_column to a column unique per row, or "
            "leave it unset so ids are dataset-qualified"
        )
    return GoldDataset(
        X=np.concatenate([dataset.X for dataset in datasets]),
        y=y,
        sample_ids=sample_ids,
        role="gold_train",
        dataset_id=dataset_id,
        feature_names=list(encoder.feature_names),
        groups=None,
        metadata={
            "pooled_from": [dataset.dataset_id for dataset in datasets],
            "raw_features": list(encoder.raw_features),
            "categorical_features": encoder.categorical_features,
        },
    ), encoder


def pool_labeled(
    paths: Sequence[str | Path], task: TaskSpec, dataset_id: str = "pooled"
) -> GoldDataset:
    """`pool_labeled_with_encoder`, for callers who only need the rows."""
    dataset, _ = pool_labeled_with_encoder(paths, task, dataset_id)
    return dataset


def _single_cell_requested(config, paths, task) -> bool:
    """Is this a single-cell task? Explicitly, or by what the table looks like.

    `auto` reads the labeled table's head: hundreds of feature columns, all of
    them non-negative integers. That is a count matrix, and running it through
    the standard encoder (median imputation and a z-score per column, on raw
    counts) is not the same analysis as the one a single-cell practitioner
    would run.
    """
    mode = getattr(config, "single_cell_preprocess", "auto")
    if mode == "on":
        return True
    if mode == "off":
        return False
    from .singlecell import looks_like_counts

    try:
        frame = read_delimited(paths[0], nrows=200)
    except Exception as e:  # noqa: BLE001 - a probe, not a decision
        logger.debug("single-cell probe could not read %s (%s)", paths[0], e)
        return False
    features = task.feature_names(frame.columns)
    return looks_like_counts(frame, features)


def check_task_against_data(frame: pd.DataFrame, task: TaskSpec) -> str:
    """Refuse a task whose type does not match the column the plan named.

    A plan can name anything as the label: the inference picked a gene (`WAS`)
    out of a single-cell panel, and because a gene is present in every
    single-cell table the role rules then called two unrelated datasets
    "labeled". The training side only found out later, as
    `Unknown label type: continuous` from inside a pseudo-label teacher -- a
    message that names neither the column nor the plan.

    Raises when the label is *also* a feature (a table cannot predict a gene
    from itself) and when a classification task has a continuous label. Both are
    refusals rather than warnings: a run that survives them produces a number
    that means nothing, which is worse than a run that stops.
    """
    column = task.target_column
    if not column or column not in frame.columns:
        return ""
    values = frame[column]
    is_numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(
        values
    )
    warning = ""
    if is_numeric and task.task_type == "classification":
        present = values.dropna()
        distinct = int(present.nunique())
        fractional = float(((present % 1) != 0).mean()) if len(present) else 0.0
        if fractional > 0.01 or distinct > max(50, len(present) // 10):
            example = ", ".join(str(value) for value in present.head(3))
            raise ValueError(
                f"the plan says this is a classification task, but its target "
                f"column {column!r} holds continuous values ({distinct:,} distinct, "
                f"e.g. {example}). Either rebuild the plan with --task-type "
                f"regression, or name the real label column instead "
                f"(--hint <the label column>) and exclude {column!r}."
            )
    features = task.feature_names(frame.columns)
    if column in features:
        raise ValueError(
            f"the target column {column!r} is also one of the {len(features):,} "
            f"feature columns, so the model would be predicting a measurement "
            f"from itself; rebuild the plan with --hint <the label column> or "
            f"exclude {column!r}"
        )
    return warning


def _run_single_cell_training(
    *,
    paths,
    task,
    supplementary_paths,
    supplementary_renames,
    output_dir,
    config,
    test_path=None,
    max_supplementary_per_dataset=None,
    max_supplementary_rows=None,
    pseudo_labeler=None,
    model_factory=None,
    model_design=None,
) -> dict:
    """Train a single-cell task on separately preprocessed sources.

    Each source is normalized and standardized on its own counts and then cut to
    the genes every source selected, so the correction sees shared measurement
    rather than one donor's scaling. The gold's labels stay as they are: this
    changes the representation, never the supervision.
    """
    from .singlecell import (
        SingleCellConfig,
        align_sources,
        describe,
        prepare_source,
        write_figures,
        write_report,
    )

    single_cell = SingleCellConfig(
        n_top_genes=int(getattr(config, "single_cell_top_genes", 2000)),
        target_sum=float(getattr(config, "single_cell_target_sum", 1e4)),
        min_cells=int(getattr(config, "single_cell_min_cells", 3)),
        seed=int(getattr(config, "seed", 42)),
    )
    gold_path = paths[0]
    with stage("reading the labeled table for preprocessing"):
        gold_frame = read_delimited(gold_path)
    # Before a minute of preprocessing: does the plan's task fit this table?
    warning = check_task_against_data(gold_frame, task)
    if warning:
        logger.warning("%s", warning)
    gold_features = task.feature_names(gold_frame.columns)
    if not task.has_target(gold_frame.columns):
        raise ValueError(
            f"{gold_path} has no target column {task.target_column!r}, so it is "
            f"not labeled data for this task"
        )
    frames: list[tuple[str, object]] = [(Path(str(gold_path)).stem, gold_frame)]
    for path in supplementary_paths:
        with stage(f"reading evidence table {Path(str(path)).name}"):
            frame = read_delimited(path)
            rename = (supplementary_renames or {}).get(str(path)) or {}
            if rename:
                frame = frame.rename(columns=dict(rename))
            if max_supplementary_per_dataset:
                frame = frame.iloc[:max_supplementary_per_dataset]
            frames.append((f"{Path(str(path)).stem}", frame))

    prepared = []
    for name, frame in frames:
        with stage(f"preprocessing {name}"):
            prepared.append(
                prepare_source(
                    frame, task.feature_names(frame.columns) or gold_features,
                    name=name, config=single_cell,
                )
            )
    panel, prepared = align_sources(
        prepared,
        max_genes=single_cell.n_top_genes,
        target_sum=single_cell.target_sum,
    )
    # Say what was done, source by source, while it is happening: this is the
    # part of a single-cell run that used to be invisible.
    logger.info(
        "single-cell preprocessing: %d source(s) on %s shared gene(s)",
        len(prepared),
        f"{len(panel):,}",
    )
    for source in prepared:
        if source.stats:
            logger.info("  %s: %s", source.name, source.stats.describe())

    # Labels and row identifiers come from the gold frame, in the same order the
    # gold matrix was built in.
    gold_labels = gold_frame[task.target_column]
    if pd.api.types.is_numeric_dtype(gold_labels) and not pd.api.types.is_bool_dtype(
        gold_labels
    ):
        y = np.asarray(gold_labels)
    else:
        y = np.asarray(gold_labels, dtype=str)
    gold = GoldDataset(
        X=prepared[0].matrix,
        y=y,
        sample_ids=task.sample_ids(gold_frame, Path(str(gold_path)).stem),
        role="gold_train",
        dataset_id=Path(str(gold_path)).stem,
        feature_names=panel,
    )
    supplementary = []
    for (name, frame), source in zip(frames[1:], prepared[1:], strict=True):
        supplementary.append(
            ExternalEvidenceDataset(
                X=source.matrix,
                sample_ids=task.sample_ids(frame, name),
                evidence_route="DIRECT_PPI",
                source_dataset_id=name,
                feature_names=panel,
                groups=None,
                audit_metadata={"preprocessing": "single_cell_per_source"},
            )
        )
    final_test = None
    if test_path:
        with stage("preprocessing the test table"):
            test_frame = read_delimited(test_path)
            test_source = prepare_source(
                test_frame,
                task.feature_names(test_frame.columns) or gold_features,
                name=Path(str(test_path)).stem,
                config=single_cell,
            )
            missing = [gene for gene in panel if gene not in set(test_source.genes)]
            if missing:
                raise ValueError(
                    f"the test table does not share {len(missing)} of the "
                    f"preprocessed gene(s), e.g. {missing[:3]}"
                )
            positions = [test_source.genes.index(gene) for gene in panel]
            test_labels = test_frame[task.target_column]
            final_test = GoldDataset(
                X=test_source.matrix[:, positions],
                y=np.asarray(
                    test_labels,
                    dtype=None
                    if pd.api.types.is_numeric_dtype(test_labels)
                    else str,
                ),
                sample_ids=task.sample_ids(test_frame, Path(str(test_path)).stem),
                role="final_test",
                dataset_id=Path(str(test_path)).stem,
                feature_names=panel,
            )

    summary = run_experiment(
        labeled=gold,
        task=task,
        supplementary=supplementary,
        test=final_test,
        output_dir=output_dir,
        config=config,
        inputs={
            "labeled_paths": [str(path) for path in paths],
            "supplementary_paths": [str(path) for path in supplementary_paths],
            "test_path": str(test_path) if test_path else None,
            "preprocessing": {
                "kind": "single_cell",
                "per_source": single_cell.to_dict(),
                "panel": panel,
                "sources": [
                    {
                        "name": source.name,
                        "cells": source.cell_count,
                        "genes_in_file": source.total_genes,
                        "genes_selected": source.n_genes,
                        "notes": source.notes,
                    }
                    for source in prepared
                ],
                "figures": [],
            },
        },
        pseudo_labeler=pseudo_labeler,
        model_factory=model_factory,
        model_design=model_design,
        max_supplementary_rows=max_supplementary_rows,
        max_supplementary_per_dataset=max_supplementary_per_dataset,
    )
    # Figures after the run: `run_experiment` refuses a directory that already
    # holds files, and a picture drawn before training is a file.
    figures: list[str] = []
    if getattr(config, "single_cell_figures", True):
        with stage("drawing the dataset figures"):
            written = write_figures(
                prepared, Path(output_dir) / "figures", labels={prepared[0].name: y}
            )
            figures = [str(path) for path in written]
    with stage("writing the preprocessing report"):
        report, record = write_report(
            prepared,
            panel,
            output_dir,
            config=single_cell,
            figures=[Path(path) for path in figures],
        )
        logger.info("preprocessing report: %s (%s)", report, record)
    account = describe(prepared, panel, single_cell)
    summary.setdefault("inputs", {}).setdefault("preprocessing", {})["figures"] = figures
    summary.setdefault("inputs", {})["preprocessing"].update(
        {
            "sources": account["sources"],
            "report": str(report),
            "record": str(record),
        }
    )
    summary["preprocessing"] = "single_cell_per_source"
    summary["preprocessing_figures"] = figures
    if figures:
        from .report import write_markdown

        write_markdown(summary, output_dir)
    return summary


def next_output_dir(path: str | Path) -> Path:
    """A fresh directory for a run: `run`, or `run-2`, `run-3`, ...

    `run_experiment` refuses to write into a directory that already holds files,
    which is the guard that keeps a re-run from silently overwriting the results
    it was meant to compare against. Re-running the same `--out` is the normal
    way to repeat a question, and inside the research loop that guard was hit
    *after* the retrieval round and an experiment-design call, then retried
    three times with the same outcome. A caller that wants a new run now gets a
    new directory, and the one that already exists is left alone.
    """
    base = Path(path)
    if not base.exists() or not any(base.iterdir()):
        return base
    for index in range(2, 1000):
        candidate = base.with_name(f"{base.name}-{index}")
        if not candidate.exists() or not any(candidate.iterdir()):
            return candidate
    raise RuntimeError(
        f"{base} and its numbered siblings are all taken; point the run at a "
        f"different directory"
    )


def run_experiment(
    *,
    labeled: GoldDataset,
    task: TaskSpec,
    output_dir: str | Path,
    supplementary: Sequence[ExternalEvidenceDataset] = (),
    test: GoldDataset | None = None,
    config: PPITrainingConfig | None = None,
    pseudo_labeler=None,
    model_factory: Callable | None = None,
    model_design: dict | None = None,
    max_supplementary_rows: int | None = None,
    max_supplementary_per_dataset: int | None = None,
    inputs: dict | None = None,
) -> dict:
    """Train on labeled data and evaluate on held-out labels.

    `test` is for the case where the evaluation set is its own labeled table
    rather than a random slice of `labeled`. Omit it and the labeled table is
    split into train / validation / test.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(
            f"output dir must be empty/new: {out} already holds "
            f"{len(list(out.iterdir()))} file(s) from an earlier run. Kosmos "
            f"refuses to write over a finished run's results: pass a new "
            f"--output-dir (or PPI_OUTPUT_DIR), move the old one aside, or use "
            f"next_output_dir() to take the next free name."
        )

    config = config or PPITrainingConfig(seed=task.seed)
    metric = task.evaluation_metric
    if task.task_type == "regression" and metric == "balanced_accuracy":
        # The task says regression and the metric was left at its
        # classification default: select on R², which is the regression
        # analogue of the balanced score the default was standing in for.
        metric = "r2"
    config = config.model_copy(
        update={
            "seed": task.seed,
            "task_type": task.task_type,
            "evaluation_metric": metric,
            "max_external_samples": max_supplementary_rows
            or config.max_external_samples,
            "max_rows_per_dataset": max_supplementary_per_dataset
            or config.max_rows_per_dataset,
        }
    )

    supplementary = list(supplementary)
    feature_names = list(labeled.feature_names)
    mirrors: list[str] = []
    gold_rows = row_fingerprint(labeled.X)
    for dataset in supplementary:
        if dataset.feature_names != feature_names:
            raise ValueError(
                f"supplementary dataset {dataset.source_dataset_id!r} has a "
                f"different feature schema from the labeled data; PPI mixes "
                f"them in one model"
            )
    kept = []
    for dataset in supplementary:
        if row_fingerprint(dataset.X) == gold_rows and len(dataset.X) == len(labeled.X):
            # The same observations, republished. Evidence has to be other rows.
            mirrors.append(dataset.source_dataset_id)
            continue
        kept.append(dataset)
    supplementary = kept

    if test is not None:
        gold_train, gold_validation = split_gold(
            labeled, validation_fraction=task.validation_fraction, seed=task.seed
        )
        final_test = test
    else:
        gold_train, gold_validation, final_test = task.split_labeled(labeled)

    pseudo = pseudo_labeler or default_pseudo_labeler(task.seed, task.task_type)
    t0 = time.time()
    kwargs = {
        "gold_train": gold_train,
        "gold_validation": gold_validation,
        "external_evidence": supplementary,
        "pseudo_labeler": pseudo,
        "config": config,
        "output_dir": str(out),
    }
    if model_factory is not None:
        kwargs["model_factory"] = model_factory
    result = run_ppi_experiment(**kwargs)
    final = evaluate_final_test(result, final_test)
    summary = {
        "mode": "ppi" if supplementary else "supervised",
        # The files this run consumed, so the summary is readable on its own.
        "inputs": inputs or {},
        "task": {
            "target_column": task.target_column,
            "task_type": task.task_type,
            "evaluation_metric": config.evaluation_metric,
            "n_features": len(feature_names),
            "classes": result.classes.tolist(),
            "description": task.description,
        },
        # Which objective trained the supplementary rows, and how hard it was
        # pushed. A run reported without this is not comparable to another one.
        "loss": {
            "mode": config.loss_mode,
            "lambda": config.ppi_lambda,
            # The gate's own knobs, recorded whenever they could have acted:
            # a run reported without them is not comparable to another one.
            **(
                {
                    "gate_kappa": config.gate_kappa,
                    "gate_scope": config.gate_scope,
                    "gate_gamma": config.gate_gamma,
                    "gate_lambda": config.gate_lambda,
                }
                if config.loss_mode == "gradient_gated"
                else {}
            ),
            "ramp_epochs": config.loss_ramp_epochs,
            "schedule": config.schedule,
            "pseudo_mode": config.pseudo_mode,
            "pseudo_targets": config.pseudo_targets,
        },
        "n_labeled": {
            "total": len(labeled.X),
            "train": len(gold_train.X),
            "validation": len(gold_validation.X),
            "test": len(final_test.X),
        },
        "supplementary": {
            "available_rows": {d.source_dataset_id: len(d.X) for d in supplementary},
            # Whether each supplementary table carried the label column. When it
            # did, the labels were ignored on purpose (multi-gold policy), and a
            # reader has to be able to see that rather than assume the cohort was
            # unlabeled.
            "sources": {
                d.source_dataset_id: {
                    "rows": len(d.X),
                    "had_target_column": bool(
                        (d.audit_metadata or {}).get("had_target_column")
                    ),
                }
                for d in supplementary
            },
            "used_rows": len(result.reproducibility["external_ids"]),
            "weight_mass": result.reproducibility["external_weight_mass"],
            # Tables that turned out to hold the labeled rows themselves. Named
            # here because "no evidence" and "evidence that was a mirror" are
            # different findings about the data that was fetched.
            "mirrors_dropped": mirrors,
        },
        # Aliases kept for the analyst prompt and existing report readers.
        "external_sources": {d.source_dataset_id: len(d.X) for d in supplementary},
        "external_used": len(result.reproducibility["external_ids"]),
        "n_gold_train": len(gold_train.X),
        "n_gold_validation": len(gold_validation.X),
        "n_final_test": len(final_test.X),
        "validation_metrics": {
            "baseline": result.baseline_metrics,
            "ppi": result.validation_metrics,
            "delta": result.delta_metrics,
        },
        "final_test_metrics": final,
        "selected_epochs": result.selected_epochs,
        # The training process itself: how many epochs each arm ran, which was
        # best, and whether early stopping fired. Per-epoch detail is in
        # `training_log.jsonl` beside this file.
        "training": getattr(result, "training", {}) or {},
        "stage_epochs": {
            "true_loss": config.stage1_epochs,
            "correction": config.stage2_epochs,
        },
        "model_design": model_design or {},
        "training_seconds": round(time.time() - t0, 1),
        "config_sha256": result.reproducibility["config_sha256"],
        "artifact_dir": str(out),
    }
    (out / "ppi_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    # A person-readable rendering of the same record, next to it.
    write_markdown(summary, out)
    return summary


def run_training(
    *,
    labeled_path: str | Path | None = None,
    labeled_paths: Sequence[str | Path] = (),
    task: TaskSpec,
    supplementary_paths: Iterable[str | Path] = (),
    output_dir: str | Path,
    test_path: str | Path | None = None,
    config: PPITrainingConfig | None = None,
    pseudo_labeler=None,
    model_factory: Callable | None = None,
    model_design: dict | None = None,
    max_supplementary_rows: int | None = None,
    max_supplementary_per_dataset: int | None = None,
    supplementary_renames: dict[str, dict[str, str]] | None = None,
) -> dict:
    """`run_experiment` for callers who have file paths rather than datasets.

    Pass one `labeled_path`, or several `labeled_paths` to pool them.

    The encoder is fitted once, on the pooled labeled rows, and then applied to
    the supplementary tables and the test table. That order matters: the
    vocabulary and the scales come from the labeled data, so a supplementary
    cohort can only be encoded the way the model it feeds was built.
    """
    paths = list(labeled_paths) or ([labeled_path] if labeled_path else [])
    if not paths:
        raise ValueError("no labeled data given: pass labeled_path or labeled_paths")
    if _single_cell_requested(config, paths, task):
        return _run_single_cell_training(
            paths=paths,
            task=task,
            supplementary_paths=list(supplementary_paths),
            supplementary_renames=supplementary_renames,
            output_dir=output_dir,
            config=config,
            test_path=test_path,
            max_supplementary_per_dataset=max_supplementary_per_dataset,
            max_supplementary_rows=max_supplementary_rows,
            pseudo_labeler=pseudo_labeler,
            model_factory=model_factory,
            model_design=model_design,
        )
    # Each of these reads a file: for a single-cell table that is a gigabyte and
    # minutes of work, which is exactly the step a "stuck" run is usually in.
    with stage(f"reading {len(paths)} labeled table(s)"):
        if len(paths) > 1:
            labeled, encoder = pool_labeled_with_encoder(paths, task)
        else:
            frame = read_table(paths[0])
            # Same check on the standard path: the failure it prevents
            # (`Unknown label type: continuous`) names neither the column nor
            # the plan.
            warning = check_task_against_data(frame, task)
            if warning:
                logger.warning("%s", warning)
            features = task.feature_names(frame.columns)
            if not task.has_target(frame.columns):
                raise ValueError(
                    f"{paths[0]} has no target column {task.target_column!r}, so it "
                    f"is not labeled data for this task"
                )
            with stage(f"encoding {len(features):,} feature column(s)"):
                encoder = FeatureEncoder.fit(frame, features)
            labeled = gold_from_frame(
                frame, task, Path(paths[0]).stem, encoder=encoder
            )
    supplementary = []
    for path in supplementary_paths:
        with stage(f"reading evidence table {Path(str(path)).name}"):
            supplementary.append(
                load_supplementary(
                    path,
                    task,
                    max_rows=max_supplementary_per_dataset,
                    reference_features=encoder.raw_features,
                    encoder=encoder,
                    rename=(supplementary_renames or {}).get(str(path)),
                )
            )
    test = (
        load_gold(test_path, task, role="final_test", encoder=encoder)
        if test_path
        else None
    )
    return run_experiment(
        labeled=labeled,
        task=task,
        supplementary=supplementary,
        test=test,
        output_dir=output_dir,
        inputs={
            "labeled_paths": [str(path) for path in paths],
            "supplementary_paths": [str(path) for path in supplementary_paths],
            "test_path": str(test_path) if test_path else None,
            # What the model actually saw, column by column: a run whose
            # provenance is a list of paths is not reproducible.
            "encoding": encoder.to_dict(),
        },
        config=config,
        pseudo_labeler=pseudo_labeler,
        model_factory=model_factory,
        model_design=model_design,
        max_supplementary_rows=max_supplementary_rows,
        max_supplementary_per_dataset=max_supplementary_per_dataset,
    )
