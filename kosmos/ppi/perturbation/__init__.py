"""Perturbation-response prediction: control + perturbation -> Δ expression.

The task contract, the three graphs, a GEARS-shaped model and three training
arms. Nothing here touches the simple per-cell pipeline; it is a second backend
on the same data layer.

    from kosmos.ppi.perturbation import (
        build_task, build_examples, perturbation_splits, split_examples,
        coexpression_graph, go_graph, run_perturbation_experiment,
    )
"""

from kosmos.ppi.perturbation.contract import (
    PerturbationExample,
    PerturbationTask,
    build_examples,
    build_task,
    parse_condition,
    perturbation_splits,
    split_examples,
    write_contract,
)
from kosmos.ppi.perturbation.graphs import (
    GO_REFERENCE,
    GraphArtifact,
    coexpression_graph,
    edge_overlap,
    go_graph,
    identity_graph,
    one_hot,
    residualize,
    write_graph_report,
)
from kosmos.ppi.perturbation.losses import deg_indices, gears_loss
from kosmos.ppi.perturbation.metrics import PerturbationMetrics, perturbation_metrics
from kosmos.ppi.perturbation.model import GearsModel, ModelConfig
from kosmos.ppi.perturbation.mlp import (
    MLP_ARMS,
    MlpConfig,
    MlpModel,
    run_mlp_arms,
)
from kosmos.ppi.perturbation.train import (
    ARMS,
    PerturbationTrainingConfig,
    run_perturbation_experiment,
)

__all__ = [
    "PerturbationTask",
    "PerturbationExample",
    "build_task",
    "build_examples",
    "parse_condition",
    "perturbation_splits",
    "split_examples",
    "write_contract",
    "GraphArtifact",
    "coexpression_graph",
    "go_graph",
    "identity_graph",
    "residualize",
    "one_hot",
    "edge_overlap",
    "write_graph_report",
    "GO_REFERENCE",
    "GearsModel",
    "ModelConfig",
    "gears_loss",
    "deg_indices",
    "perturbation_metrics",
    "PerturbationMetrics",
    "run_perturbation_experiment",
    "PerturbationTrainingConfig",
    "ARMS",
    "MLP_ARMS",
    "MlpModel",
    "MlpConfig",
    "run_mlp_arms",
]
