"""What task are we training, and how does a table become an example of it.

This is the layer that was missing: the PPI engine always spoke in terms of
`X` / `y` / `sample_ids`, while the flow that fed it spoke in terms of
`cell_type` / `ENSG*` / `DonorID`. A task spec is the adapter between the two,
written once, so no dataset-specific column name lives in the training code.

Two decisions are deliberately *not* part of this type:

  * **No grouping, no leakage prevention.** Roles are decided by whether a
    table carries the task's label column: labeled tables are gold, unlabeled
    tables are supplementary evidence. Group columns are not consulted, not
    required, and not recorded as a split axis.
  * **No distribution assumptions.** Where the supplementary rows came from is
    provenance (the dataset id), not a split dimension. Keeping the two apart
    is what lets any task use any table.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .schemas import GoldDataset
from .split import stratified_split


@dataclass(frozen=True)
class TaskSpec:
    """The task: which column is predicted, from which features, scored how.

    Feature selection is by explicit list, by prefix, or by "everything that is
    not the target / not an identifier". Prefixes are the convenient form for a
    wide table (a few thousand `ENSG*` columns, `gene_*`, `cg*`), explicit lists
    are the reproducible form, and both can be combined: explicit columns win,
    prefixes filter what remains.
    """

    #: The label column. `None` means "this spec is for unlabeled tables only".
    target_column: str | None = None
    #: Exact feature columns, in order. Validated against the table.
    feature_columns: tuple[str, ...] | None = None
    #: Keep columns starting with any of these (applied when feature_columns is None).
    feature_prefixes: tuple[str, ...] = ()
    #: Never features: identifiers, metadata, other label columns.
    exclude_columns: tuple[str, ...] = ()
    #: Column holding stable per-row identifiers. Defaults to `<dataset>:<row>`.
    sample_id_column: str | None = None

    task_type: Literal["classification", "regression"] = "classification"
    evaluation_metric: str = "balanced_accuracy"

    #: Fractions of the LABELED table: test first, then validation out of the rest.
    test_fraction: float = 0.2
    validation_fraction: float = 0.2
    seed: int = 42
    #: Free-form task description, carried into provenance (not interpreted).
    description: str = ""

    def has_target(self, columns: Sequence[str]) -> bool:
        return self.target_column is not None and self.target_column in set(columns)

    def feature_names(self, columns: Sequence[str]) -> list[str]:
        """The ordered feature list this spec selects from a table's columns."""
        present = list(columns)
        present_set = set(present)
        if self.feature_columns is not None:
            missing = [c for c in self.feature_columns if c not in present_set]
            if missing:
                raise ValueError(
                    f"task spec names {len(missing)} feature column(s) the table "
                    f"does not have, e.g. {missing[:3]}"
                )
            return list(self.feature_columns)
        excluded = set(self.exclude_columns)
        if self.target_column:
            excluded.add(self.target_column)
        if self.sample_id_column:
            excluded.add(self.sample_id_column)
        names = [c for c in present if c not in excluded]
        if self.feature_prefixes:
            names = [c for c in names if c.startswith(tuple(self.feature_prefixes))]
        if not names:
            raise ValueError(
                "no feature columns selected; check feature_prefixes / "
                "exclude_columns against the table's columns"
            )
        return names

    def sample_ids(self, frame: pd.DataFrame, dataset_id: str) -> np.ndarray:
        if self.sample_id_column:
            if self.sample_id_column not in frame.columns:
                raise ValueError(
                    f"sample_id_column {self.sample_id_column!r} is not in the table"
                )
            return frame[self.sample_id_column].astype(str).to_numpy()
        # Dataset-qualified, so two tables can be pooled without colliding.
        return np.array([f"{dataset_id}:{i}" for i in range(len(frame))], dtype=str)

    def split_labeled(
        self, dataset: GoldDataset
    ) -> tuple[GoldDataset, GoldDataset, GoldDataset]:
        """Split labeled data into train / validation / test, at random.

        Test is carved out first so the reported number is untouched by model
        selection; validation is carved out of what remains. Stratified for
        classification when the classes allow it.
        """
        if dataset.role != "gold_train":
            raise ValueError("only a labeled (gold) dataset can be split here")
        n = len(dataset.X)
        if n < 3:
            raise ValueError(
                f"labeled data has {n} row(s); at least 3 are needed for a "
                f"train/validation/test split"
            )
        indices = np.arange(n)
        if self.task_type == "classification":
            # Class-aware at both steps, and it keeps at least one row of every
            # class in training: a random split can otherwise leave a two-row
            # class entirely in validation, which the trainer refuses ("classes
            # absent from gold training") after the evidence has been read.
            train_val, test = stratified_split(dataset.y, self.test_fraction, self.seed)
            inner_train, inner_val = stratified_split(
                dataset.y[train_val], self.validation_fraction, self.seed + 1
            )
            train, validation = train_val[inner_train], train_val[inner_val]
        else:
            train_val, test = train_test_split(
                indices, test_size=self.test_fraction, random_state=self.seed
            )
            train, validation = train_test_split(
                train_val,
                test_size=self.validation_fraction,
                random_state=self.seed + 1,
            )

        def subset(idx, role) -> GoldDataset:
            return GoldDataset(
                X=dataset.X[idx],
                y=dataset.y[idx],
                sample_ids=dataset.sample_ids[idx],
                role=role,
                dataset_id=dataset.dataset_id,
                feature_names=dataset.feature_names,
                groups=None,
                metadata={
                    "task": self.description or self.target_column or "",
                    "split": f"random seed={self.seed}",
                    "parent_dataset_id": dataset.dataset_id,
                },
            )

        return (
            subset(train, "gold_train"),
            subset(validation, "gold_validation"),
            subset(test, "final_test"),
        )


def task_spec_from_config(
    config: Mapping[str, Any], *, description: str = ""
) -> TaskSpec | None:
    """Build a task from a run configuration, or None when none is declared.

    This is the single place the CLI's flags become a task. Nothing else in the
    training path reads a config key, so adding a task field is an edit here and
    in the CLI -- not a hunt through the director.
    """
    target = config.get("task_target_column")
    if not target:
        return None
    feature_columns = config.get("task_feature_columns")
    return TaskSpec(
        target_column=str(target),
        feature_columns=tuple(feature_columns) if feature_columns else None,
        feature_prefixes=tuple(config.get("task_feature_prefixes") or ()),
        exclude_columns=tuple(config.get("task_exclude_columns") or ()),
        sample_id_column=config.get("task_sample_id_column"),
        task_type=str(config.get("task_type", "classification")),
        evaluation_metric=str(config.get("ppi_evaluation_metric", "balanced_accuracy")),
        seed=int(config.get("ppi_seed", 42)),
        validation_fraction=float(config.get("task_validation_fraction", 0.2)),
        test_fraction=float(config.get("task_test_fraction", 0.2)),
        description=description,
    )
