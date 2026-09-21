"""Classification-first prediction-powered training, independent of evidence modality."""

from .features import FeatureEncoder, UnencodableColumn, max_cardinality
from .flow import (
    load_gold,
    load_supplementary,
    next_output_dir,
    pool_labeled,
    pool_labeled_with_encoder,
    run_experiment,
    run_training,
)
from .gating import GateStats, GradientGate
from .integration import PPIEvidenceTrainer, RoutingPredictor
from .losses import PPILoss, PseudoLabelLoss, SupervisedLoss
from .pseudo_labeler import PretrainedPseudoLabeler, prepare_pseudo_labeler
from .schemas import (
    ExternalEvidenceDataset,
    GoldDataset,
    PPITrainingConfig,
    PPITrainingResult,
    PseudoLabeledExternal,
    PseudoLabeledGold,
)
from .split import split_gold
from .task_spec import TaskSpec, task_spec_from_config
from .trainer import evaluate_final_test, run_ppi_experiment, train_ppi

__all__ = [
    "PPIEvidenceTrainer",
    "RoutingPredictor",
    "PPILoss",
    "PseudoLabelLoss",
    "SupervisedLoss",
    "GradientGate",
    "GateStats",
    "TaskSpec",
    "task_spec_from_config",
    "load_gold",
    "load_supplementary",
    "pool_labeled",
    "pool_labeled_with_encoder",
    "FeatureEncoder",
    "UnencodableColumn",
    "max_cardinality",
    "run_experiment",
    "run_training",
    "next_output_dir",
    "PretrainedPseudoLabeler",
    "prepare_pseudo_labeler",
    "ExternalEvidenceDataset",
    "GoldDataset",
    "PPITrainingConfig",
    "PPITrainingResult",
    "PseudoLabeledExternal",
    "PseudoLabeledGold",
    "split_gold",
    "evaluate_final_test",
    "run_ppi_experiment",
    "train_ppi",
]
