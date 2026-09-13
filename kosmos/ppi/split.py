"""Development splitting and identity checks; final-test evaluation is separate."""

from dataclasses import replace

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .schemas import ExternalEvidenceDataset, GoldDataset


def split_gold(dataset: GoldDataset, validation_fraction=0.2, seed=42):
    if type(dataset) is not GoldDataset or dataset.role != "gold_train":
        raise ValueError("Only gold_train can be split for development")
    indices = np.arange(len(dataset.X))
    if dataset.groups is not None:
        train, val = next(
            GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed).split(
                dataset.X, dataset.y, dataset.groups
            )
        )
    else:
        train, val = train_test_split(
            indices, test_size=validation_fraction, random_state=seed, stratify=dataset.y
        )

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
