"""Gold-only cross-fitting or reviewed frozen pretrained classifiers."""

import copy
from hashlib import sha256
from typing import Protocol

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .schemas import PseudoLabeledGold


class PseudoLabeler(Protocol):
    def fit(self, X, y): ...
    def predict(self, X): ...
    def predict_proba(self, X): ...


class PretrainedPseudoLabeler:
    """An already trained task classifier, not a foundation embedding alone.

    Caller review must address target domain, features, ontology and training
    overlap. No fit call is made. Optional known training IDs are checked too.
    """

    def __init__(
        self,
        estimator,
        *,
        model_reference,
        code_reference,
        feature_names,
        applicability_evidence,
        leakage_review,
        training_sample_ids=(),
    ):
        if (
            not model_reference
            or not code_reference
            or not applicability_evidence
            or not leakage_review
        ):
            raise ValueError("Pretrained model requires version, applicability and leakage review")
        self.estimator = estimator
        self.model_reference = model_reference
        self.code_reference = code_reference
        self.feature_names = feature_names
        self.applicability_evidence = list(applicability_evidence)
        self.leakage_review = leakage_review
        self.training_sample_ids = set(training_sample_ids)

    def fit(self, X, y):
        raise ValueError("Frozen pretrained classifiers must not be fit by the experiment")

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    @property
    def classes_(self):
        return self.estimator.classes_


def fingerprint(gold):
    digest = sha256(gold.X.tobytes())
    for values in (gold.y, gold.sample_ids, gold.groups):
        if values is not None:
            digest.update(repr(np.asarray(values).tolist()).encode())
    digest.update(repr(gold.feature_names).encode())
    digest.update(gold.dataset_id.encode())
    return digest.hexdigest()


def predictions(estimator, X, classes):
    y = np.asarray(estimator.predict(X))
    if y.shape != (len(X),) or not set(y).issubset(set(classes)):
        raise ValueError("Pseudo predictions must align with rows and target classes")
    prob = None
    probability_estimator = (
        estimator.estimator if isinstance(estimator, PretrainedPseudoLabeler) else estimator
    )
    if hasattr(probability_estimator, "predict_proba"):
        prob = np.asarray(estimator.predict_proba(X), dtype=float)
        model_classes = np.asarray(estimator.classes_)
        if prob.shape != (len(X), len(model_classes)):
            raise ValueError("Invalid pseudo-label probability shape")
        if set(model_classes) != set(classes) or len(model_classes) != len(classes):
            raise ValueError("Pseudo classifier class ontology does not match gold")
        prob = prob[:, [list(model_classes).index(c) for c in classes]]
        if (
            prob.shape != (len(X), len(classes))
            or not np.isfinite(prob).all()
            or (prob < 0).any()
            or (prob > 1).any()
            or not np.allclose(prob.sum(1), 1)
        ):
            raise ValueError("Invalid pseudo-label probabilities")
    return y, prob


class PreparedPseudoLabeler:
    def __init__(self, models, classes, gold, pseudo_gold, provenance):
        self.models, self.classes = models, classes
        self.gold_fingerprint = fingerprint(gold)
        self.gold_dataset_id = gold.dataset_id
        self.pseudo_gold = pseudo_gold
        self.provenance = provenance
        self.model_reference = provenance["model_reference"]
        self.code_reference = provenance["code_reference"]

    def predict_with_probabilities(self, X):
        outputs = [predictions(m, X, self.classes) for m in self.models]
        if all(p is not None for _, p in outputs):
            prob = np.mean([p for _, p in outputs], axis=0)
            return self.classes[prob.argmax(1)], prob
        votes = np.stack([y for y, _ in outputs])
        counts = np.stack([(votes == c).sum(0) for c in self.classes], axis=1)
        return self.classes[counts.argmax(1)], None


def prepare_pseudo_labeler(gold, estimator, config):
    if gold.role != "gold_train":
        raise ValueError("Pseudo-label fitting requires gold_train")
    if isinstance(estimator, PreparedPseudoLabeler):
        if estimator.gold_fingerprint != fingerprint(gold):
            raise ValueError("Prepared pseudo-labeler belongs to different gold training data")
        if estimator.provenance["settings"] != pseudo_settings(config):
            raise ValueError("Prepared pseudo-labeler settings differ from experiment config")
        return estimator
    classes = np.unique(gold.y)
    if len(classes) < 2:
        raise ValueError("Classification requires at least two gold training classes")
    models, folds = [], []
    probabilities = None
    if config.pseudo_mode == "pretrained":
        if not isinstance(estimator, PretrainedPseudoLabeler):
            raise ValueError("Pretrained mode requires a reviewed PretrainedPseudoLabeler")
        if estimator.feature_names != gold.feature_names:
            raise ValueError("Pretrained feature schema does not match gold")
        if estimator.training_sample_ids.intersection(gold.sample_ids):
            raise ValueError("Pretrained training samples overlap gold correction examples")
        model = copy.deepcopy(estimator)
        y_pred, probabilities = predictions(model, gold.X, classes)
        models.append(model)
        provenance = {
            "model_reference": estimator.model_reference,
            "code_reference": estimator.code_reference,
            "applicability_evidence": estimator.applicability_evidence,
            "leakage_review": estimator.leakage_review,
            "known_training_sample_ids": sorted(estimator.training_sample_ids),
        }
    else:
        if isinstance(estimator, PretrainedPseudoLabeler):
            raise ValueError("Reviewed pretrained model requires pseudo_mode='pretrained'")

        def fit(indices, fold):
            model = clone(estimator)
            seeds = {
                k: config.seed + fold
                for k in model.get_params(deep=True)
                if k.endswith("random_state")
            }
            model.set_params(**seeds)
            model.fit(gold.X[indices], gold.y[indices])
            return model

        if config.pseudo_mode == "in_sample":
            model = fit(np.arange(len(gold.X)), 0)
            y_pred, probabilities = predictions(model, gold.X, classes)
            models.append(model)
        else:
            if gold.groups is None:
                splitter = StratifiedKFold(
                    config.cross_fit_folds, shuffle=True, random_state=config.seed
                )
            else:
                splitter = StratifiedGroupKFold(
                    config.cross_fit_folds, shuffle=True, random_state=config.seed
                )
            y_pred = np.empty_like(gold.y)
            covered = np.zeros(len(gold.X), dtype=int)
            for fold, (train, holdout) in enumerate(splitter.split(gold.X, gold.y, gold.groups)):
                if set(gold.y[train]) != set(classes):
                    raise ValueError(
                        "Every cross-fit training fold must contain every target class"
                    )
                model = fit(train, fold)
                pred, prob = predictions(model, gold.X[holdout], classes)
                models.append(model)
                y_pred[holdout], covered[holdout] = pred, covered[holdout] + 1
                if prob is not None:
                    if probabilities is None:
                        probabilities = np.zeros((len(gold.X), len(classes)))
                    probabilities[holdout] = prob
                folds.append(
                    {
                        "train_ids": gold.sample_ids[train].tolist(),
                        "heldout_ids": gold.sample_ids[holdout].tolist(),
                    }
                )
            if not np.all(covered == 1):
                raise ValueError(
                    "Cross-fit predictions must cover every gold observation exactly once"
                )
        provenance = {
            "model_reference": f"{type(estimator).__module__}.{type(estimator).__name__}",
            "code_reference": "kosmos.ppi.pseudo_labeler/v1",
            "estimator_parameters": repr(estimator.get_params(deep=True)),
        }
    provenance.update(
        settings=pseudo_settings(config),
        folds=folds,
        gold_prediction_mode=config.pseudo_mode,
        external_prediction_mode="fold_ensemble" if len(models) > 1 else "single_model",
    )
    if config.pseudo_targets == "soft" and probabilities is None:
        raise ValueError("Soft pseudo targets require predict_proba")
    return PreparedPseudoLabeler(
        models,
        classes,
        gold,
        PseudoLabeledGold(
            gold.X.copy(), gold.y.copy(), y_pred, probabilities, gold.sample_ids.copy()
        ),
        provenance,
    )


def pseudo_settings(config):
    return {
        k: getattr(config, k) for k in ("seed", "pseudo_mode", "cross_fit_folds", "pseudo_targets")
    }
