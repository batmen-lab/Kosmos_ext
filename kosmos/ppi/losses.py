"""Isolated fixed-lambda signed CE correction, inspired by scDesign3/src/train_ppi.py.

Default schedule alternates true and correction stages. A joint objective is
also explicit; neither claims to reproduce scDesign's learnable lambda.
"""

import torch
import torch.nn.functional as F


def supervised_per_row(logits, target, *, task_type="classification", label_smoothing=0.0):
    """One loss value per row, for either head, hard labels or soft targets.

    The same measurement the two objectives above are built from, exposed so the
    gradient gate can ask for a *single* row's loss -- and so a synthetic loss
    and a gold loss are the same quantity, which is what makes their gradients
    comparable at all.
    """
    if task_type == "regression":
        values = logits.squeeze(-1) if logits.dim() > 1 else logits
        return (values - target.reshape(-1)) ** 2
    if target.dim() == 2:
        # A teacher's probabilities: -sum_c p_c log softmax_c.
        return -(target * F.log_softmax(logits, dim=1)).sum(dim=1)
    return F.cross_entropy(
        logits, target, reduction="none", label_smoothing=label_smoothing
    )


class SupervisedLoss:
    """`L = ℓ(gold) + λ·ℓ(synthetic)`: ordinary supervision on both.

    This is the naive baseline the gradient gate is compared against, and the
    source of the two terms it gates. Nothing is subtracted and nothing is
    reweighted beyond `coefficient`: whatever the synthetic rows say is applied
    in full, which is exactly the behaviour the gate is meant to replace.

    Hard labels and a teacher's probabilities are both accepted, so a run can
    hold everything else fixed and vary only which labels the synthetic rows
    carry.
    """

    def __init__(self, coefficient=1.0, label_smoothing=0.0, task_type="classification"):
        self.coefficient = coefficient
        self.label_smoothing = label_smoothing
        self.task_type = task_type

    def per_row(self, logits, target):
        return supervised_per_row(
            logits,
            target,
            task_type=self.task_type,
            label_smoothing=self.label_smoothing if target.dim() == 1 else 0.0,
        )

    def labeled_term(self, logits, target):
        return self.per_row(logits, target).mean()

    def external_term(self, logits, target, weights=None, population_size=None):
        per_row = self.per_row(logits, target)
        if weights is None:
            return per_row.mean()
        return (per_row * weights).sum() / weights.sum().clamp_min(1e-12)

    def total_loss(
        self,
        gold_logits,
        y_true,
        y_pseudo,
        external_logits=None,
        external_pseudo=None,
        external_weights=None,
        population_size=0,
        weight_mass=0.0,
    ):
        true = self.labeled_term(gold_logits, y_true)
        ext = torch.zeros_like(true)
        if external_logits is not None and external_weights is not None:
            ext = self.external_term(
                external_logits, external_pseudo, external_weights, population_size
            )
        correction = self.coefficient * ext
        return {
            "loss_true": true,
            "loss_pseudo_gold": torch.zeros_like(true),
            "loss_correction": correction,
            "loss_external": ext,
            "loss_total": true + correction,
        }


class PPILoss:
    """The signed correction. Unbiased, and therefore has a negative term.

    The algebra is the same for a classifier and a regressor: only the per-row
    loss differs (cross-entropy over class logits, or squared error on a single
    predicted value). `task_type` is set from the config by the trainer.
    """

    def __init__(self, coefficient=1.0, label_smoothing=0.0, task_type="classification"):
        self.coefficient = coefficient
        self.label_smoothing = label_smoothing
        self.task_type = task_type

    def _per_row(self, logits, target):
        """One loss value per row, for either head."""
        if self.task_type == "regression":
            values = logits.squeeze(-1) if logits.dim() > 1 else logits
            return (values - target.reshape(-1)) ** 2
        smoothing = self.label_smoothing if target.dim() == 1 else 0.0
        return F.cross_entropy(
            logits, target, reduction="none", label_smoothing=smoothing
        )

    def labeled_term(self, logits, target):
        return self._per_row(logits, target).mean()

    def external_term(self, logits, target, weights, population_size):
        # Uniform external minibatches give an unbiased estimate of sum_j w_j L_j.
        per_row = self._per_row(logits, target)
        return (per_row * weights).mean() * population_size

    def total_loss(
        self,
        gold_logits,
        y_true,
        y_pseudo,
        external_logits=None,
        external_pseudo=None,
        external_weights=None,
        population_size=0,
        weight_mass=0.0,
    ):
        true = self.labeled_term(gold_logits, y_true)
        pseudo = self.labeled_term(gold_logits, y_pseudo)
        ext = torch.zeros_like(true)
        correction = torch.zeros_like(true)
        if external_logits is not None and weight_mass > 0:
            ext = self.external_term(
                external_logits, external_pseudo, external_weights, population_size
            )
            correction = self.coefficient * (ext - weight_mass * pseudo)
        return {
            "loss_true": true,
            "loss_pseudo_gold": pseudo,
            "loss_correction": correction,
            "loss_external": ext,
            "loss_total": true + correction,
        }


class PseudoLabelLoss:
    """Supervise the unlabeled rows with the teacher's *probabilities*. No negative term.

        L = CE(gold, y)  +  λ(t) · Σᵢ wᵢ·H(pᵢ, θ(xᵢᵉˣᵗ)) / Σᵢ wᵢ

    where `H(p, ·)` is the cross-entropy of the model against the teacher's
    softmax output `p`, and `λ(t)` ramps from 0 to `lambda` over
    `ramp_epochs`.

    Three deliberate differences from `PPILoss`, and each has a cost:

    * **There is no subtraction.** The signed correction's second term is what
      makes the external rows' pseudo-label errors cancel in expectation. This
      loss keeps them, so it is biased: it is self-training, not PPI, and the
      claim it supports is empirical ("it helped on the held-out split"), not
      estimator-theoretic.
    * **The external term is a weighted mean, not a population-scaled sum.**
      The sum exists upstream to put the two terms of the correction on one
      scale; without a negative term it would make this loss scale with the
      number of unlabeled rows.
    * **The targets are soft.** A teacher's uncertainty carries information --
      the point of the run is often the rare classes, which are exactly where
      an argmax throws that information away.

    The negative term's *other* effect goes away too: it trained the model to
    disagree with its own teacher on the labeled rows. Nothing here does that.
    """

    def __init__(
        self,
        coefficient=1.0,
        label_smoothing=0.0,
        ramp_epochs=0,
        task_type="classification",
    ):
        self.coefficient = coefficient
        self.label_smoothing = label_smoothing
        self.task_type = task_type
        self.ramp_epochs = max(0, int(ramp_epochs))
        #: The value currently in force; equals `coefficient` unless ramping.
        self.effective_coefficient = float(coefficient) if self.ramp_epochs == 0 else 0.0

    # -- schedule -----------------------------------------------------------

    def schedule_epoch(self, epoch: int, max_epochs: int | None = None) -> float:
        """Set the coefficient for this epoch, ramping from 0 when asked.

        A ramp exists because the external term is now ordinary supervision: at
        epoch 1 the model is untrained, so its own errors and the teacher's
        would be learned together. Starting at zero lets the labeled rows decide
        the initial direction first.
        """
        if self.ramp_epochs <= 0:
            self.effective_coefficient = float(self.coefficient)
        else:
            progress = min(1.0, max(0.0, epoch) / float(self.ramp_epochs))
            self.effective_coefficient = float(self.coefficient) * progress
        return self.effective_coefficient

    # -- terms --------------------------------------------------------------

    def labeled_term(self, logits, target):
        if self.task_type == "regression":
            values = logits.squeeze(-1) if logits.dim() > 1 else logits
            return ((values - target.reshape(-1)) ** 2).mean()
        smoothing = self.label_smoothing if target.dim() == 1 else 0.0
        return F.cross_entropy(logits, target, label_smoothing=smoothing)

    def external_term(self, logits, target_probs, weights, population_size=None):
        """The teacher's per-row loss: soft cross-entropy, or squared error.

        A regression teacher predicts a value, not a distribution, so its
        "soft target" is that value and the loss against it is the same squared
        error used on the labeled rows.
        """
        if self.task_type == "regression":
            if target_probs is None:
                raise ValueError(
                    "PseudoLabelLoss needs the teacher's predicted values for a "
                    "regression task (pseudo_targets='soft')"
                )
            values = logits.squeeze(-1) if logits.dim() > 1 else logits
            per_row = (values - target_probs.reshape(-1)) ** 2
            if weights is None:
                return per_row.mean()
            return (per_row * weights).sum() / weights.sum().clamp_min(1e-12)
        if target_probs is None:
            raise ValueError(
                "PseudoLabelLoss supervises with the teacher's probabilities, so "
                "external pseudo targets must be soft (pseudo_targets='soft')"
            )
        per_row = -(target_probs * F.log_softmax(logits, dim=1)).sum(dim=1)
        if weights is None:
            return per_row.mean()
        total = weights.sum()
        return (per_row * weights).sum() / total.clamp_min(1e-12)

    def total_loss(
        self,
        gold_logits,
        y_true,
        y_pseudo,
        external_logits=None,
        external_pseudo=None,
        external_weights=None,
        population_size=0,
        weight_mass=0.0,
    ):
        true = self.labeled_term(gold_logits, y_true)
        ext = torch.zeros_like(true)
        if external_logits is not None and external_weights is not None:
            ext = self.external_term(
                external_logits, external_pseudo, external_weights, population_size
            )
        # `loss_pseudo_gold` and `loss_correction` are reported for continuity
        # with the signed loss's history; neither exists in this objective.
        correction = self.effective_coefficient * ext
        return {
            "loss_true": true,
            "loss_pseudo_gold": torch.zeros_like(true),
            "loss_correction": correction,
            "loss_external": ext,
            "loss_total": true + correction,
        }
