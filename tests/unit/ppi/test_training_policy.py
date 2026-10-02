"""The shared training policy: one schedule and one correction for every task."""

from __future__ import annotations

import torch

from kosmos.ppi.training import correction_stage, signed_correction


def test_the_two_stage_predicate_cycles_like_the_column_task():
    stages = [
        correction_stage(
            epoch, use_external=True, schedule="two_stage", stage1_epochs=4, stage2_epochs=1
        )
        for epoch in range(1, 11)
    ]
    assert stages == [False, False, False, False, True] * 2
    assert not correction_stage(
        5, use_external=False, schedule="two_stage", stage1_epochs=4, stage2_epochs=1
    )
    assert not correction_stage(
        5, use_external=True, schedule="joint", stage1_epochs=4, stage2_epochs=1
    )


def test_the_signed_correction_is_the_one_algebra_both_backends_share():
    gold = torch.tensor(2.0)
    pseudo_gold = torch.tensor(1.0)
    extension = torch.tensor(3.0)
    terms = signed_correction(
        gold=gold, pseudo_gold=pseudo_gold, extension=extension, coefficient=0.5, mass=1.0
    )

    assert float(terms["loss_true"]) == 2.0
    assert float(terms["loss_correction"]) == 0.5 * (3.0 - 1.0)
    assert float(terms["loss_total"]) == 2.0 + 0.5 * (3.0 - 1.0)
    assert set(terms) == {
        "loss_true",
        "loss_pseudo_gold",
        "loss_external",
        "loss_correction",
        "loss_total",
    }
