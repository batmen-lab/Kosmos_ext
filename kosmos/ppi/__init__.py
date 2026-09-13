"""Classification-first prediction-powered training, independent of evidence modality."""

from .integration import PPIEvidenceTrainer, RoutingPredictor
from .losses import PPILoss
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
from .trainer import evaluate_final_test, run_ppi_experiment, train_ppi

__all__ = [
    "PPIEvidenceTrainer",
    "RoutingPredictor",
    "PPILoss",
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
