"""Budgeted external selection and pseudo-label materialization."""

import numpy as np

from .schemas import PseudoLabeledExternal


def prepare_external(datasets, labeler, config):
    nonempty = sorted([d for d in datasets if len(d.X)], key=lambda d: d.source_dataset_id)
    capacities = [min(len(d.X), config.max_rows_per_dataset) for d in nonempty]
    quotas = [0] * len(nonempty)
    remaining = min(sum(capacities), config.max_external_samples)
    # Balanced source quotas keep a large source from consuming the row budget.
    while remaining:
        for i, capacity in enumerate(capacities):
            if quotas[i] < capacity and remaining:
                quotas[i] += 1
                remaining -= 1
    rng = np.random.default_rng(config.seed)
    result = []
    for data, count in zip(nonempty, quotas, strict=False):
        if not count:
            continue
        idx = np.sort(rng.choice(len(data.X), count, replace=False))
        pred, prob = labeler.predict_with_probabilities(data.X[idx])
        raw = data.evidence_weight
        source_mass = (
            config.external_weight_budget / len(nonempty) if raw is None else float(raw.sum())
        )
        weights = np.ones(count) if raw is None else raw[idx].copy()
        # Preserve upstream source mass after row capping, before optional alpha.
        if weights.sum() > 0:
            weights *= source_mass / weights.sum()
        if config.use_evidence_weights and data.alpha is not None:
            weights *= data.alpha[idx]
        audit = {
            "source_dataset_id": data.source_dataset_id,
            "evidence_route": data.evidence_route,
            "audit": data.audit_metadata,
            "selected_indices": idx.tolist(),
            "alpha": data.alpha[idx].tolist() if data.alpha is not None else None,
            "source_labels": (
                data.source_labels[idx].tolist() if data.source_labels is not None else None
            ),
        }
        result.append(
            PseudoLabeledExternal(
                data.X[idx].copy(), pred, prob, data.sample_ids[idx].copy(), audit, weights
            )
        )
    mass = sum(d.weights.sum() for d in result)
    if mass > config.external_weight_budget:
        for data in result:
            data.weights *= config.external_weight_budget / mass
    return result
