"""summary.md: the run's numbers written the way a person reads them."""

from __future__ import annotations

import numpy as np

from kosmos.ppi.report import gate_table, render_markdown, write_markdown


def test_the_gate_gets_its_own_table_in_the_summary():
    """A gated run reports what the gate did, epoch by epoch."""
    payload = summary()
    payload["loss"] = {
        "mode": "gradient_gated",
        "lambda": 1.0,
        "ramp_epochs": 0,
        "schedule": "joint",
        "pseudo_mode": "cross_fit",
        "pseudo_targets": "hard",
        "gate_kappa": 1.0,
        "gate_scope": "batch",
        "gate_gamma": 1.0,
        "gate_lambda": 1.0,
    }
    payload["training"] = {
        "gate": {
            "scope": "batch",
            "gradient_cosine_mean": 0.42,
            "gradient_cosine_std": 0.11,
            "synthetic_weight_mean": 0.31,
            "synthetic_weight_std": 0.05,
            "synthetic_active_fraction": 0.75,
            "epochs": [
                {
                    "epoch": 1,
                    "gradient_cosine": 0.51,
                    "gradient_cosine_std": 0.10,
                    "gold_grad_norm": 3.2,
                    "synthetic_grad_norm": 4.1,
                    "synthetic_weight": 0.33,
                    "synthetic_weight_std": 0.04,
                    "synthetic_active_fraction": 0.80,
                    "train_gold_loss": 0.71,
                    "train_synthetic_loss": 1.02,
                    "validation_balanced_accuracy": 0.74,
                }
            ],
        }
    }

    rendered = render_markdown(payload, ".")
    assert "gradient_gated" in rendered
    assert "g = g_G + max(0, cos" in rendered
    table = gate_table(payload)
    assert any("cos(g_G, g_S)" in line for line in table)
    assert any("| 1 |" in line for line in table)
    # A run without gate diagnostics says so rather than inventing a table.
    assert "No gate diagnostics" in "\n".join(gate_table({"training": {}}))


def summary(**overrides) -> dict:
    base = {
        "mode": "ppi",
        "task": {
            "target_column": "cell_type",
            "task_type": "classification",
            "evaluation_metric": "balanced_accuracy",
            "n_features": 1500,
            "classes": ["B", "T", "NK"],
            "description": "demo task",
        },
        "loss": {
            "mode": "plain",
            "lambda": 0.5,
            "ramp_epochs": 3,
            "schedule": "joint",
            "pseudo_mode": "cross_fit",
            "pseudo_targets": "soft",
        },
        "n_labeled": {"total": 100, "train": 60, "validation": 20, "test": 20},
        "supplementary": {"available_rows": {"external-a": 500}, "used_rows": 400, "weight_mass": 0.5},
        "external_used": 400,
        "inputs": {"labeled_paths": ["labeled.csv"], "supplementary_paths": ["external.csv"]},
        "validation_metrics": {
            "baseline": {"accuracy": 0.80, "balanced_accuracy": 0.70, "macro_f1": 0.71},
            "ppi": {"accuracy": 0.79, "balanced_accuracy": 0.72, "macro_f1": 0.73},
        },
        "final_test_metrics": {
            "baseline": {"accuracy": 0.75, "balanced_accuracy": 0.65, "macro_f1": 0.66},
            "ppi": {"accuracy": 0.74, "balanced_accuracy": 0.67, "macro_f1": 0.68},
        },
        "selected_epochs": {"baseline": 5, "ppi": 6},
        "stage_epochs": {"true_loss": 4, "correction": 1},
        "training_seconds": 12.3,
        "config_sha256": "abc123",
        "artifact_dir": "run",
    }
    base.update(overrides)
    return base


def test_markdown_states_the_task_the_loss_and_both_splits(tmp_path):
    text = render_markdown(summary(), tmp_path)
    assert "# Training run summary" in text
    assert "`cell_type`" in text
    assert "mode: `plain`, lambda: `0.5`, ramp: `3`" in text
    assert "**biased** by construction" in text  # honesty about what plain is
    assert "## Validation (used for model selection)" in text
    assert "## Final test (untouched by selection)" in text
    assert "| balanced_accuracy | 0.7000 | 0.7200 | +0.0200 |" in text
    assert "`labeled.csv`" in text and "`external.csv`" in text
    assert "config sha256: `abc123`" in text


def test_signed_runs_say_the_estimator_is_unbiased(tmp_path):
    text = render_markdown(
        summary(loss={"mode": "signed", "lambda": 1, "ramp_epochs": 0, "schedule": "two_stage"}),
        tmp_path,
    )
    assert "unbiased" in text
    # ("unbiased" contains "biased", so assert on the sentence, not the word.)
    assert "**biased** by construction" not in text


def test_a_run_without_a_loss_field_says_so(tmp_path):
    text = render_markdown(summary(loss=None), tmp_path)
    assert "not recorded" in text
    assert "signed" in text


def test_supervised_runs_say_there_is_no_supplementary_data(tmp_path):
    text = render_markdown(
        summary(mode="supervised", supplementary={"available_rows": {}}, external_used=0),
        tmp_path,
    )
    assert "this is plain supervised training" in text


def test_per_class_section_uses_the_validation_predictions(tmp_path):
    classes = ["rare", "mid", "common"]
    rows = 60
    y = np.array([0] * 2 + [1] * 8 + [2] * 50)
    baseline = np.zeros((rows, 3))
    augmented = np.zeros((rows, 3))
    baseline[:, 2] = 1.0  # everything predicted as `common`
    augmented[:, 1] = 1.0
    augmented[:8, 1] = 1.0
    np.savez_compressed(
        tmp_path / "validation_predictions.npz",
        sample_ids=np.array([f"s{i}" for i in range(rows)]),
        y_true=np.array([classes[i] for i in y]),
        baseline_probability=baseline,
        ppi_probability=augmented,
    )
    text = render_markdown(summary(task={**summary()["task"], "classes": classes}), tmp_path)
    assert "Rarest 3 of 3 classes" in text
    assert "| rare | 2 |" in text
    assert "Mean delta" in text


def test_the_section_says_when_predictions_are_missing(tmp_path):
    text = render_markdown(summary(), tmp_path)
    assert "No per-class breakdown" in text


def test_numeric_labels_are_matched_as_text(tmp_path):
    """A numeric label column (a quality score) must not empty the table."""
    classes = [3, 4, 5, 6, 7, 8]
    rows = 60
    y = np.array([5] * 40 + [6] * 15 + [3] * 5)
    baseline = np.zeros((rows, len(classes)))
    augmented = np.zeros((rows, len(classes)))
    baseline[:, 2] = 1.0  # always predicts 5
    augmented[:, 2] = 1.0
    augmented[40:55, 3] = 1.0  # predicts 6 correctly
    np.savez_compressed(
        tmp_path / "validation_predictions.npz",
        sample_ids=np.array([f"s{i}" for i in range(rows)]),
        y_true=y,
        baseline_probability=baseline,
        ppi_probability=augmented,
    )
    text = render_markdown(summary(task={**summary()["task"], "classes": classes}), tmp_path)
    assert "Rarest 3 of 3 classes" in text
    assert "Mean delta" in text  # the crash was here before the fix


def test_a_supplementary_table_that_had_labels_says_so(tmp_path):
    """Demoted gold: the reader must see that labels existed and were ignored."""
    summary_with_source = summary(
        supplementary={
            "available_rows": {"winequality-red": 1599},
            "sources": {"winequality-red": {"rows": 1599, "had_target_column": True}},
            "used_rows": 1599,
            "weight_mass": 0.5,
        }
    )
    text = render_markdown(summary_with_source, tmp_path)
    assert "carries the label column" in text
    assert "ignored by policy" in text


def test_write_markdown_lands_beside_the_json(tmp_path):
    path = write_markdown(summary(), tmp_path)
    assert path.name == "summary.md" and path.parent == tmp_path
    assert path.read_text(encoding="utf-8").startswith("# Training run summary")
