"""Serializable contracts for non-gold evidence; unknowns remain explicit."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", str_min_length=1)


class TargetDataProfile(Record):
    prediction_task: str
    label_ontology: str
    observation_unit: str
    organism: str
    biological_system: str
    condition: str
    experimental_context: str
    target_modality: str
    feature_names: list[str] = Field(min_length=1)
    preprocessing: str
    gold_dataset_id: str
    biological_covariates: dict[str, Any] = Field(default_factory=dict)
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_features(self):
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Target feature names must be unique")
        return self


class Assessment(Record):
    status: Literal["accept", "reject", "uncertain"]
    rationale: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)
    # Explicit assessments may cover non-identical domains. Metadata equality
    # alone never establishes biological relevance.
    factors: dict[str, Any] = Field(default_factory=dict)


class SourceLabel(Record):
    available: bool = False
    column: str | None = None
    ontology: str | None = None
    audit_compatible: bool = False
    compatibility_evidence: list[str] = Field(default_factory=list)
    usable_as_gold: Literal[False] = False
    reason: str = "Non-gold source annotations are auxiliary audit evidence only"

    @model_validator(mode="after")
    def validate_label(self):
        if self.available and not self.column:
            raise ValueError("Available source labels require a column")
        if self.audit_compatible and not (
            self.available and self.ontology and self.compatibility_evidence
        ):
            raise ValueError("Label auditing requires labels, ontology and compatibility evidence")
        return self


class CandidateDataset(Record):
    """Manifest around an actual dataset retrieved by Kosmos, not a search hit."""

    dataset_id: str
    source: str
    file_path: str
    source_modality: str
    observation_unit: str
    feature_names: list[str] = Field(min_length=1)
    preprocessing: str
    biological_relevance: Assessment
    provenance: list[str] = Field(min_length=1)
    organism: str | None = None
    biological_system: str | None = None
    condition: str | None = None
    assay: str | None = None
    publication: str | None = None
    sample_size: int | None = Field(default=None, gt=0)
    processing_level: str | None = None
    pairing_information: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_label: SourceLabel = Field(default_factory=SourceLabel)
    quality_components: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_features(self):
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Source features must be unique")
        if self.source_label.column in self.feature_names:
            raise ValueError("Source label cannot be an input feature")
        return self


class TransformSpec(Record):
    method_id: str
    kind: Literal["harmonize", "translate"]
    source_modality: str
    target_modality: str
    # Explicitly reviewed dataset applicability; not inferred from modality alone.
    applicable_dataset_ids: list[str] = Field(min_length=1)
    target_profile: TargetDataProfile
    evidence_for_applicability: list[str] = Field(min_length=1)
    code_reference: str
    model_reference: str
    training_domain: str
    pairing: Literal["paired", "unpaired", "not_applicable"]
    validation_metrics: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


class RoutingRecord(Record):
    candidate: CandidateDataset
    target: TargetDataProfile
    route: Literal["DIRECT_PPI", "TRANSLATE_PPI", "REJECT"]
    decision: Literal["accept", "reject", "defer"]
    reason: str
    translation_required: bool
    feature_compatibility: dict[str, Any]
    transform: TransformSpec | None = None
    pseudo_label_strategy: str = "Predict using a frozen model trained only on trusted gold data"
    audit_signals: dict[str, Any] = Field(default_factory=dict)
    uncertainties: list[str] = Field(default_factory=list)
    quality_components: dict[str, Any] = Field(default_factory=dict)
    execution: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(Record):
    split_role: Literal["validation", "cross_validation"]
    split_id: str
    protocol: str
    metric: str
    higher_is_better: bool = True
    baseline: float = Field(allow_inf_nan=False)
    augmented: float = Field(allow_inf_nan=False)

    @property
    def delta(self) -> float:
        """Raw change; improvement accounts for metric direction separately."""
        return self.augmented - self.baseline


class PPIConfig(Record):
    implementation: str
    code_reference: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    # Aggregate evidence influence, irrespective of external dataset size.
    external_weight_budget: float = Field(default=0.5, gt=0, le=1)
    max_rows_per_dataset: int = Field(default=10000, gt=0)
    seed: int = 42
