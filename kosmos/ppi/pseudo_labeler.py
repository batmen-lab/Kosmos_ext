"""Gold-only cross-fitting or reviewed frozen pretrained classifiers."""

import copy
import logging
from collections import Counter
from hashlib import sha256

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from .schemas import PseudoLabeledGold

logger = logging.getLogger(__name__)


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


def predictions(estimator, X, classes, task_type="classification"):
    if task_type == "regression":
        values = np.asarray(estimator.predict(X), dtype=float).reshape(-1)
        if values.shape != (len(X),) or not np.isfinite(values).all():
            raise ValueError(
                "A regression teacher must predict one finite value per row"
            )
        return values, None
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
    def __init__(self, models, classes, gold, pseudo_gold, provenance, task_type="classification"):
        self.models, self.classes = models, classes
        self.task_type = task_type
        self.gold_fingerprint = fingerprint(gold)
        self.gold_dataset_id = gold.dataset_id
        self.pseudo_gold = pseudo_gold
        self.provenance = provenance
        self.model_reference = provenance["model_reference"]
        self.code_reference = provenance["code_reference"]

    def predict_with_probabilities(self, X):
        outputs = [
            predictions(m, X, self.classes, self.task_type) for m in self.models
        ]
        if self.task_type == "regression":
            # The fold models predict values; their mean is the teacher's
            # prediction, and there is no distribution to call "soft".
            stacked = np.stack([values for values, _ in outputs])
            return stacked.mean(axis=0), None
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
    regression = getattr(config, "task_type", "classification") == "regression"
    classes = None if regression else np.unique(gold.y)
    if regression:
        if len(gold.y) < 2:
            raise ValueError("Regression requires at least two gold training rows")
    elif len(classes) < 2:
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
        y_pred, probabilities = predictions(
            model, gold.X, classes, "regression" if regression else "classification"
        )
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
            y_pred, probabilities = predictions(
                model, gold.X, classes, "regression" if regression else "classification"
            )
            models.append(model)
        elif not regression and min(Counter(gold.y.tolist()).values()) < 2:
            # A class with one row cannot be held out and kept at the same time,
            # and `StratifiedKFold` will not try. Rather than refuse the run --
            # a single-cell table's rarest cell type is exactly what the question
            # is about -- the teacher is trained on all the labeled rows, as
            # `in_sample` does, and the provenance says so.
            logger.warning(
                "the rarest of %d classes has a single row, so no cross-fit fold "
                "can hold it out; the pseudo-label teacher is fitted in-sample "
                "instead (pseudo_mode=in_sample)",
                len(classes),
            )
            model = fit(np.arange(len(gold.X)), 0)
            y_pred, probabilities = predictions(
                model, gold.X, classes, "classification"
            )
            models.append(model)
        else:
            if regression:
                # There are no strata to keep: KFold still holds every row out
                # exactly once, which is the property cross-fitting needs.
                splitter = (
                    GroupKFold(config.cross_fit_folds)
                    if gold.groups is not None
                    else KFold(config.cross_fit_folds, shuffle=True, random_state=config.seed)
                )
            elif gold.groups is None:
                # `StratifiedKFold` needs at least one row of every class in
                # every fold. A single-cell table has 45 cell types over 2,000
                # cells and several of them are rarer than that, so the fold
                # count comes down to what the rarest class can carry. Below two
                # there is no cross-fit to do at all -- the teacher then sees all
                # the labeled rows, which is what `in_sample` means.
                counts = Counter(gold.y.tolist())
                usable = min([config.cross_fit_folds, *counts.values()])
                splitter = StratifiedKFold(
                    usable, shuffle=True, random_state=config.seed
                )
                if usable < config.cross_fit_folds:
                    logger.info(
                        "cross-fit folds reduced from %d to %d: the rarest class has "
                        "%d row(s)",
                        config.cross_fit_folds,
                        usable,
                        min(counts.values()),
                    )
            else:
                splitter = StratifiedGroupKFold(
                    config.cross_fit_folds, shuffle=True, random_state=config.seed
                )
            y_pred = (
                np.zeros(len(gold.X), dtype=float)
                if regression
                else np.empty_like(gold.y)
            )
            covered = np.zeros(len(gold.X), dtype=int)
            for fold, (train, holdout) in enumerate(splitter.split(gold.X, gold.y, gold.groups)):
                if not regression and set(gold.y[train]) != set(classes):
                    raise ValueError(
                        "Every cross-fit training fold must contain every target class"
                    )
                model = fit(train, fold)
                pred, prob = predictions(
                    model, gold.X[holdout], classes, "regression" if regression else "classification"
                )
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
    if config.pseudo_targets == "soft" and probabilities is None and not regression:
        raise ValueError("Soft pseudo targets require predict_proba")
    return PreparedPseudoLabeler(
        models,
        classes,
        gold,
        PseudoLabeledGold(
            gold.X.copy(), gold.y.copy(), y_pred, probabilities, gold.sample_ids.copy()
        ),
        provenance,
        "regression" if regression else "classification",
    )


def pseudo_settings(config):
    return {
        k: getattr(config, k) for k in ("seed", "pseudo_mode", "cross_fit_folds", "pseudo_targets")
    }
