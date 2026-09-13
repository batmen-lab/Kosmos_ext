# Non-gold evidence routing prototype

`kosmos.evidence` is an opt-in boundary between datasets already retrieved by
Kosmos and a user-supplied pseudo-label/PPI implementation. It leaves the gold
retrieval and training path intact. It does not yet automate scientific search,
metadata extraction from publications, relevance review or translator discovery.
The self-contained `kosmos.ppi` module now implements the training side through
the `PPITrainer` integration contract. See [PPI training](ppi_training.md) for
the loss, gold/external pseudo predictions, baseline, pretrained support and
cross-fitting behavior.

## Inputs and integration

### ResearchWorkflow integration

The evidence stage is now an explicit, opt-in stage of
`kosmos.workflow.research_loop.ResearchWorkflow`. Legacy runs remain unchanged
when no evidence arguments are supplied. To enable it, provide an
`EvidencePipeline`, pseudo-label predictor, PPI trainer/config, and either
retrieved `CandidateDataset` manifests or an `evidence_discoverer` callback:

```python
workflow = ResearchWorkflow(
    research_objective="Classify BMMC cell states from target RNA features",
    anthropic_client=client,
    artifacts_dir="artifacts/research-bmmc",
    evidence_pipeline=pipeline,
    evidence_candidates=retrieved_candidates,
    evidence_predictor=routing_predictor,
    evidence_trainer=ppi_trainer,
    evidence_config=routing_config,
    evidence_output_dir="artifacts/research-bmmc/evidence",
)
result = await workflow.run(num_cycles=3, tasks_per_cycle=5)
```

For dynamic retrieval, replace `evidence_candidates` with a callback receiving
`(research_objective, cycle, context)` and returning fully populated manifests.
This is the narrow integration point for a repository/API adapter; it must not
return bare search hits:

```python
def discover(objective, cycle, context):
    return existing_kosmos_dataset_adapter.search_and_retrieve(
        objective, cycle=cycle, context=context
    )

workflow = ResearchWorkflow(
    research_objective=objective,
    evidence_pipeline=pipeline,
    evidence_discoverer=discover,
    evidence_predictor=routing_predictor,
    evidence_trainer=ppi_trainer,
    evidence_config=routing_config,
)
```

By default the stage runs once after the requested final research cycle. Set
`evidence_every_cycle=True` to run after every cycle. Runs are written to
`evidence_output_dir/cycle-NNN`; cycle results, final statistics and reports
include accepted/deferred/rejected counts and PPI evaluations. The repository
still does not provide a domain-specific dataset download/search adapter, so a
scientific connector must supply `evidence_discoverer` for real retrieval.

1. Construct `TargetDataProfile` from the research objective and gold dataset.
   Specify ordered feature names, normalization, observation unit, ontology and
   biological inclusion/exclusion criteria. The profile is general to patients,
   samples, cells and other observation units. Do not infer these from a filename.
2. Wrap every retrieved non-gold file in `CandidateDataset` (or use
   `CandidateDataset.model_validate(retrieved_manifest)`). Preserve repository,
   persistent accession (`dataset_id`), publication, assay, organism, system,
   sample count, processing level, pairing and original metadata. Record relevant
   population, intervention, time and task assessments in
   `biological_relevance.factors`, with evidence references and rationale.
   A metadata match alone is insufficient: acceptance requires an evidence-backed
   review. Missing/uncertain relevance defers execution. Rejection is retained.
3. Register only reviewed `Transform(spec, callable)` implementations.
   A spec binds code/model versions, supporting literature or pairing evidence,
   training domain, validation metrics, limitations, candidate accessions and the
   complete target profile. Dataset applicability is an explicit human/scientific
   assessment, not an assertion proved by the software. Modality names should be
   canonicalized upstream (comparison is case-insensitive). Multimodal datasets
   should identify the selected view and retain pairing in their manifest.
4. Supply a frozen `Predictor` and a `PPITrainer` adapter. The built-in
   `RoutingPredictor`/`PPIEvidenceTrainer` pair supports gold-only cross-fitting
   or a reviewed pretrained classifier without changing evidence routes.
5. Call `EvidencePipeline.run(...)` with an empty experiment output directory.

```python
from kosmos.evidence import CandidateDataset, EvidencePipeline, PPIConfig, TargetDataProfile

# These manifests are produced from the scientific objective, gold metadata,
# and the outputs of the existing Kosmos retrieval workflow.
target = TargetDataProfile.model_validate(target_manifest)
candidates = [CandidateDataset.model_validate(item) for item in retrieved_manifests]
pipeline = EvidencePipeline(target, transforms=reviewed_transforms)
result = pipeline.run(
    candidates=candidates,
    predictor=frozen_gold_predictor,
    trainer=existing_ppi_adapter,
    config=PPIConfig(
        implementation="your-ppi-implementation/version",
        code_reference="your-repository@commit",
        parameters={"validation_protocol": "patient-disjoint development split"},
        external_weight_budget=0.5,
        max_rows_per_dataset=10000,
        seed=42,
    ),
    output_dir="artifacts/evidence/experiment-001",
    # Required only if several reviewed transformations apply to a candidate.
    methods={"accession-needing-selection": "reviewed-method-id"},
)
```

## Routing and representation

- `DIRECT_PPI`: observed target-modality features, optionally harmonized by a
  reviewed adapter. Different feature order, preprocessing or observation unit
  requires explicit harmonization; missing features are never silently imputed.
- `TRANSLATE_PPI`: a reviewed cross-modal transform produces target features.
  No applicable translator means `route=REJECT, decision=defer`; explicit
  biological rejection means `decision=reject`. Multiple translators also defer
  until selected. Translated data retain their distinct representation marker.
- Source labels never decide the route and cannot be designated gold. They are
  excluded from input features and retained separately. Agreement is computed
  only with an explicitly reviewed compatible ontology. No automatic ontology
  mapping or replacement of predictions is performed.

Files are loaded through the existing `DataProvider` with synthetic fallback
**disabled**. The first implementation supports numeric tabular target features.
Transforms must preserve row identity and produce the exact ordered target
feature schema. Aggregation that changes observation units/row counts requires
an upstream provenance-preserving dataset preparation step. Missing/non-finite
features, invalid probabilities and row reordering fail preparation.

## Adapter responsibilities

`Predictor` exposes `gold_dataset_id`, `model_reference`, `code_reference` and
`predict(DataFrame) -> Prediction`. Return one label/value per row; classification
can additionally return probabilities with class ordering. Entropy and margin
are computed from those probabilities. Calibration, OOD and ensemble diagnostics
are retained when supplied and remain unknown otherwise. Regression can omit
class probabilities. A matching gold dataset ID is a declared lineage check;
the module cannot inspect whether an arbitrary external model was really trained
only on that dataset. With `RoutingPredictor`, this ID binds the gold correction
dataset; the separately recorded provenance distinguishes a gold-trained model
from an independently pretrained one.

`PPITrainer.evaluate(target, evidence, config) -> ValidationResult` owns the actual
PPI algorithm and access to gold/development data. Keep source labels auxiliary,
respect each batch's `weights`, and compare gold-only and augmented models using
the same metric and validation split. The weights sum to the configured budget
across **all** external rows; each source gets equal total mass, independent of
size. This is a conservative prototype policy, not learned evidence quality.
The adapter must map this budget to its estimator without renormalizing each
source to full influence. The wrapper cannot enforce internals of arbitrary
external training code.

No final test dataset is passed through this API, and `ValidationResult` rejects
test split roles. Use patient/group/time separation where appropriate and avoid
training/validation overlap in gold, translator and external datasets. These
split and lineage assertions need verification by the supplied adapter; the
generic routing layer does not independently detect patient overlap or test data
falsely marked as validation. The built-in PPI adapter adds global sample/group
overlap checks when the required identity metadata is supplied.

## Outputs and experimental loop

- `run.json`: target profile, complete candidate manifests, rejected/deferred
  decisions, transform provenance, file SHA-256, sampled row identifiers,
  predictor lineage, probability/uncertainty audits, PPI settings and evaluations.
- `evidence_summary.csv`: dataset relevance, modalities, translation requirement,
  translator, label availability/audit usability, route, quality components,
  uncertainty, decision and rationale.
- `evidence_N_features.csv`, `evidence_N_pseudo_labels.csv` and, when available,
  `evidence_N_source_labels.csv`: distinct standardized artifacts. Probability
  arrays are saved in `run.json`; per-source weight mass is in each evaluation.

Each accepted source is evaluated independently, then all accepted sources as a
combination. `delta` is augmented minus baseline performance; `improvement`
accounts for lower-is-better metrics. Failed preparation and trainer errors are
recorded without silently claiming success. Use validation utility for later
manual routing/translator decisions. Automated replanning, learned confidence,
combinatorial source selection and final-test evaluation remain outside scope.

Run the offline contract tests:

```bash
python -m pytest tests/unit/evidence -q -o addopts=''
```

Tests use explicitly artificial predictor/transform/trainer fixtures. Their
reported deltas validate bookkeeping only and are not scientific performance
claims or credible cross-modal translator examples.
