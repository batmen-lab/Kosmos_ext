"""Isolated fixed-lambda signed CE correction, inspired by scDesign3/src/train_ppi.py.

Default schedule alternates true and correction stages. A joint objective is
also explicit; neither claims to reproduce scDesign's learnable lambda.
"""

import torch
import torch.nn.functional as F


class PPILoss:
    def __init__(self, coefficient=1.0, label_smoothing=0.0):
        self.coefficient = coefficient
        self.label_smoothing = label_smoothing

    def labeled_term(self, logits, target):
        return F.cross_entropy(logits, target, label_smoothing=self.label_smoothing)

    def external_term(self, logits, target, weights, population_size):
        # Uniform external minibatches give an unbiased estimate of sum_j w_j CE_j.
        per_row = F.cross_entropy(
            logits, target, reduction="none", label_smoothing=self.label_smoothing
        )
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
