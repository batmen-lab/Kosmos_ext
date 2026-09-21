"""Development splitting and identity checks; final-test evaluation is separate."""

import logging
from dataclasses import replace

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from .schemas import ExternalEvidenceDataset, GoldDataset

logger = logging.getLogger(__name__)


def split_gold(dataset: GoldDataset, validation_fraction=0.2, seed=42):
    if type(dataset) is not GoldDataset or dataset.role != "gold_train":
        raise ValueError("Only gold_train can be split for development")
    if dataset.groups is not None:
        train, val = next(
            GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed).split(
                dataset.X, dataset.y, dataset.groups
            )
        )
    else:
        train, val = stratified_split(dataset.y, validation_fraction, seed)

    def subset(idx, role):
        return replace(
            dataset,
            X=dataset.X[idx],
            y=dataset.y[idx],
            sample_ids=dataset.sample_ids[idx],
            groups=dataset.groups[idx] if dataset.groups is not None else None,
            role=role,
        )

    return subset(train, "gold_train"), subset(val, "gold_validation")


def stratified_split(
    labels: np.ndarray, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """A split where every class in validation was also seen in training.

    `train_test_split(stratify=...)` does not promise that: with a two-row class
    both rows can land in validation, and the training loop then refuses the
    split -- "Validation contains classes absent from gold training". In a
    single-cell table (45 cell types over 2,000 cells) the rare classes are
    exactly the ones the question is about, so each class gives up rows
    proportionally and keeps at least one for training.
    """
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    held_out_nothing: list[str] = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        rng.shuffle(members)
        wanted = int(round(len(members) * validation_fraction))
        # At least one row of every class stays in training: a class the model
        # has never seen is not a validation case, it is a different problem.
        take = min(wanted, max(0, len(members) - 1))
        if take == 0:
            held_out_nothing.append(str(label))
        validation_indices.extend(int(index) for index in members[:take])
        train_indices.extend(int(index) for index in members[take:])
    if held_out_nothing:
        logger.info(
            "%d class(es) were too small to hold a row out for validation, e.g. "
            "%s; they train and are not scored",
            len(held_out_nothing),
            held_out_nothing[:3],
        )
    return (
        np.sort(np.asarray(train_indices, dtype=int)),
        np.sort(np.asarray(validation_indices, dtype=int)),
    )


def validate_roles(train, validation, external):
    if type(train) is not GoldDataset or train.role != "gold_train":
        raise ValueError("Training requires observed gold_train")
    if type(validation) is not GoldDataset or validation.role != "gold_validation":
        raise ValueError("Model selection requires gold_validation, never final_test")
    # Revalidate mutable dataclass fields at the public boundary.
    train.__post_init__()
    validation.__post_init__()
    seen = set(train.sample_ids)
    if seen.intersection(validation.sample_ids):
        raise ValueError("Gold training/validation sample overlap")
    seen.update(validation.sample_ids)
    if (train.groups is None) != (validation.groups is None):
        raise ValueError("Group metadata must be supplied on both gold roles")
    val_groups = set(validation.groups) if validation.groups is not None else set()
    if train.groups is not None and val_groups.intersection(train.groups):
        raise ValueError("Gold training/validation group overlap")
    source_ids = set()
    for data in [validation, *external]:
        if data.X.shape[1] != train.X.shape[1] or data.feature_names != train.feature_names:
            raise ValueError("All data must share the exact target feature schema")
    for data in external:
        if type(data) is not ExternalEvidenceDataset:
            raise ValueError("Only ExternalEvidenceDataset is accepted as external evidence")
        data.__post_init__()
        if data.source_dataset_id in source_ids or data.source_dataset_id == train.dataset_id:
            raise ValueError("Duplicate evidence source or gold dataset presented as evidence")
        source_ids.add(data.source_dataset_id)
        if seen.intersection(data.sample_ids):
            raise ValueError("External sample overlap with gold/validation/other evidence")
        seen.update(data.sample_ids)
        if val_groups and data.groups is None:
            raise ValueError("External group metadata required for group-aware validation")
        if data.groups is not None and val_groups.intersection(data.groups):
            raise ValueError("External evidence overlaps validation groups")
