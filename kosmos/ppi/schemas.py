"""Explicit data roles and serializable classification experiment configuration."""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


def matrix(X, ids, names):
    X = np.asarray(X, dtype=np.float32)
    ids = np.asarray(ids, dtype=str)
    if X.ndim != 2 or not X.shape[1] or not np.isfinite(X).all():
        raise ValueError("X must be a finite numeric matrix with at least one feature")
    if ids.shape != (len(X),) or len(set(ids)) != len(ids) or np.any(ids == ""):
        raise ValueError("sample_ids must be unique nonempty identifiers aligned to X")
    if names is not None and (len(names) != X.shape[1] or len(set(names)) != len(names)):
        raise ValueError("feature_names must uniquely identify every column in order")
    return X.copy(), ids.copy()


@dataclass
class GoldDataset:
    X: np.ndarray
    y: np.ndarray
    sample_ids: np.ndarray
    role: Literal["gold_train", "gold_validation", "final_test"]
    dataset_id: str
    feature_names: list[str] | None = None
    groups: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    representation: Literal["observed"] = "observed"

    def __post_init__(self):
        self.X, self.sample_ids = matrix(self.X, self.sample_ids, self.feature_names)
        self.y = np.asarray(self.y)
        if not len(self.X) or self.y.shape != (len(self.X),):
            raise ValueError("Gold data must have one true label per observation and be nonempty")
        if self.y.dtype.kind not in "biufUS" or (
            self.y.dtype.kind in "f" and not np.isfinite(self.y).all()
        ):
            raise ValueError("Labels must be finite numbers or strings")
        if self.role not in {"gold_train", "gold_validation", "final_test"}:
            raise ValueError("Invalid gold role")
        if not self.dataset_id or self.representation != "observed":
            raise ValueError("Gold must be trusted observed data with a dataset identifier")
        if (
            self.metadata.get("evidence_route")
            or self.metadata.get("representation", "observed") != "observed"
        ):
            raise ValueError("External/translated evidence cannot enter GoldDataset")
        if self.groups is not None:
            self.groups = np.asarray(self.groups, dtype=str)
            if self.groups.shape != (len(self.X),) or np.any(self.groups == ""):
                raise ValueError("Group identifiers must align with observations")


@dataclass
class ExternalEvidenceDataset:
    X: np.ndarray
    sample_ids: np.ndarray
    evidence_route: Literal["DIRECT_PPI", "TRANSLATE_PPI"]
    source_dataset_id: str
    feature_names: list[str] | None = None
    alpha: np.ndarray | None = None
    # Absolute upstream mass, distinct from alpha. Never normalized back to one.
    evidence_weight: np.ndarray | None = None
    groups: np.ndarray | None = None
    audit_metadata: dict[str, Any] = field(default_factory=dict)
    source_labels: np.ndarray | None = None
    role: Literal["external_evidence"] = "external_evidence"

    def __post_init__(self):
        self.X, self.sample_ids = matrix(self.X, self.sample_ids, self.feature_names)
        if self.role != "external_evidence" or self.evidence_route not in {
            "DIRECT_PPI",
            "TRANSLATE_PPI",
        }:
            raise ValueError("External evidence requires an external role and accepted route")
        if not self.source_dataset_id:
            raise ValueError("Evidence requires a source dataset ID")
        for name in ("alpha", "evidence_weight"):
            values = getattr(self, name)
            if values is not None:
                values = np.asarray(values, dtype=float)
                if (
                    values.shape != (len(self.X),)
                    or not np.isfinite(values).all()
                    or (values < 0).any()
                    or (values > 1).any()
                ):
                    raise ValueError(f"{name} must contain aligned values in [0,1]")
                setattr(self, name, values)
        for name in ("groups", "source_labels"):
            values = getattr(self, name)
            if values is not None:
                values = np.asarray(values)
                if values.shape != (len(self.X),):
                    raise ValueError(f"{name} must align with observations")
                setattr(self, name, values)


@dataclass
class PseudoLabeledGold:
    X: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray | None
    sample_ids: np.ndarray


@dataclass
class PseudoLabeledExternal:
    X: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray | None
    sample_ids: np.ndarray
    evidence_metadata: dict[str, Any]
    weights: np.ndarray


CLASSIFICATION_LEARNING_RATE = 0.001
#: A regressor optimizes squared error against a standardized target, where the
#: useful weights are of order one: at the classifier's rate it spends an epoch
#: budget moving a few hundredths. Callers who set a rate explicitly keep it.
REGRESSION_LEARNING_RATE = 0.01


class PPITrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    task_type: Literal["classification", "regression"] = "classification"
    seed: int = 42
    gold_batch_size: int = Field(default=64, gt=0)
    external_batch_size: int = Field(default=128, gt=0)
    learning_rate: float = Field(default=CLASSIFICATION_LEARNING_RATE, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    max_epochs: int = Field(default=50, gt=0)
    patience: int = Field(default=10, gt=0)
    min_delta: float = Field(default=0, ge=0)
    pseudo_mode: Literal["cross_fit", "in_sample", "pretrained"] = "cross_fit"
    cross_fit_folds: int = Field(default=5, ge=2)
    pseudo_targets: Literal["hard", "soft"] = "hard"
    #: Which objective trains the external rows.
    #:   signed             -- PPI: +external, −gold pseudo loss (unbiased, negative term)
    #:   plain              -- pseudo-label distillation: +external only (self-training)
    #:   gold_plus_synthetic -- L_G + λ·L_S: the naive baseline the gate replaces
    #:   gradient_gated     -- g_G + λ_t·g_S, λ_t = max(0, cos(g_G, g_S))·min(1, κ‖g_G‖/‖g_S‖)
    loss_mode: Literal[
        "signed", "plain", "gold_plus_synthetic", "gradient_gated"
    ] = "signed"
    #: How loud the accepted synthetic gradient may be, as a multiple of the
    #: gold gradient's norm. 1.0 = never louder than gold.
    gate_kappa: float = Field(default=1.0, ge=0)
    #: `batch` gates one synthetic gradient per step; `sample` gates each
    #: synthetic row against the gold gradient (one backward per row).
    gate_scope: Literal["batch", "sample"] = "batch"
    #: Exponent on each row's agreement when `gate_scope="sample"`.
    gate_gamma: float = Field(default=1.0, ge=0)
    #: The outer λ on the accepted synthetic gradient. 1.0 = the gate's own
    #: weight is the whole story.
    gate_lambda: float = Field(default=1.0, ge=0)
    #: Epochs over which `plain` ramps its coefficient from 0. 0 = constant.
    loss_ramp_epochs: int = Field(default=0, ge=0)
    use_evidence_weights: bool = False
    external_weight_budget: float = Field(default=0.5, ge=0, le=1)
    max_external_samples: int = Field(default=10000, ge=0)
    max_rows_per_dataset: int = Field(default=10000, gt=0)
    ppi_lambda: float = Field(default=1, ge=0, le=1)
    #: Single-cell preprocessing, per source: counts -> HVG -> log1p ->
    #: per-gene z-score -> negatives clipped to 0. `auto` applies it when the
    #: labeled table looks like counts (hundreds+ of non-negative integer
    #: feature columns); `off` leaves every table to the standard encoder.
    single_cell_preprocess: Literal["auto", "on", "off"] = "auto"
    single_cell_top_genes: int = Field(default=2000, gt=0)
    single_cell_target_sum: float = Field(default=1e4, gt=0)
    single_cell_min_cells: int = Field(default=3, ge=1)
    #: Write the preprocessing figures beside the run's summary.
    single_cell_figures: bool = True
    schedule: Literal["two_stage", "joint"] = "two_stage"
    stage1_epochs: int = Field(default=4, gt=0)
    stage2_epochs: int = Field(default=1, gt=0)
    stage2_lr_multiplier: float = Field(default=0.1, gt=0)
    label_smoothing: float = Field(default=0, ge=0, lt=1)
    evaluation_metric: Literal[
        # Classification: higher is better.
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_auroc",
        "macro_auprc",
        # Regression: `mse`/`mae`/`rmse` are the errors (lower is better), and
        # `r2`/`pearson` the agreement (higher is better). Selection uses max(),
        # so the negated errors are the selectable forms of the error metrics.
        "mse",
        "mae",
        "rmse",
        "neg_mse",
        "neg_mae",
        "r2",
        "pearson",
    ] = "balanced_accuracy"
    model_reference: str = "torch-linear-v1"

    @model_validator(mode="after")
    def _loss_mode_is_consistent(self):
        if self.loss_mode in {"plain", "gold_plus_synthetic", "gradient_gated"}:
            # None of these has a negative term to alternate with. The two-stage
            # schedule exists to keep the signed correction's two objectives
            # apart; here there is one objective, so `joint` is the honest value.
            if self.schedule != "joint":
                object.__setattr__(self, "schedule", "joint")
        if self.task_type == "regression":
            # A regression run has one sensible default per field, and the
            # caller who left the classification defaults in place meant these.
            if self.evaluation_metric == "balanced_accuracy":
                object.__setattr__(self, "evaluation_metric", "r2")
            if self.learning_rate == CLASSIFICATION_LEARNING_RATE:
                object.__setattr__(self, "learning_rate", REGRESSION_LEARNING_RATE)
            if self.loss_mode == "plain" and self.schedule != "joint":
                object.__setattr__(self, "schedule", "joint")
            return self
        if self.loss_mode == "plain":
            if self.pseudo_targets != "soft":
                raise ValueError(
                    "loss_mode='plain' supervises the external rows with the "
                    "teacher's probabilities, so pseudo_targets must be 'soft' "
                    "(set PPI_PSEUDO_TARGETS=soft)"
                )
        return self


@dataclass
class PPITrainingResult:
    model: Any
    baseline_model: Any
    classes: np.ndarray
    trained_model_path: str
    baseline_model_path: str
    training_history: dict[str, list[dict]]
    validation_metrics: dict[str, Any]
    baseline_metrics: dict[str, Any]
    delta_metrics: dict[str, float | None]
    n_gold: int
    n_external_available: int
    n_external_used: int
    evidence_sources_used: list[str]
    reproducibility: dict[str, Any]
    selected_epochs: dict[str, int]
    pseudo_gold: PseudoLabeledGold
    pseudo_external: list[PseudoLabeledExternal]
    #: The training process: epochs run, best epoch, why it stopped, per arm.
    training: dict[str, Any] = field(default_factory=dict)

    def summary(self):
        excluded = {"model", "baseline_model", "classes", "pseudo_gold", "pseudo_external"}
        return {
            **{k: v for k, v in vars(self).items() if k not in excluded},
            "classes": self.classes.tolist(),
        }
