# Two-stage data workflow: fetch → review → experiment

`kosmos run` answers a question end to end: it fetches data, plans it, and
trains, all in one process. When a person wants to look at the data **before**
anything trains, use the two standalone stages instead. The handoff between
them is a single file — `plan.json` — so either can be run, inspected and
re-run without the other.

```
kosmos dataprep   "question"  ──►  data package (plan.json, data.md, ...)
                                        │
                                        ▼   human review / edit plan.json
kosmos experiment "question"  ──►  run/ (summary.md, metrics.json, figures/)
                --data-plan <package>/plan.json
```

## Stage 1 — fetch (never trains)

```bash
.venv/bin/python -m kosmos.cli.main dataprep \
  "Does knocking out the gene STAT1 with CRISPR change the transcriptome of human macrophages?" \
  --task perturbation --domain biology \
  --out artifacts/dataprep/stat1
```

Writes into `--out` (default `artifacts/dataprep/<slug>-<stamp>`):

| file | what it is |
|---|---|
| `data.md` | **the human-facing review document** — the task the fetcher read, the gold / supplementary / refused tables, and what to check |
| `plan.json` | the machine contract stage 2 reads: target/condition column, gold, supplementary, features |
| `data_report.md` | the full reasoning chain (every search, listing, download and refusal) |
| `data_report.json`, `reviews.json`, `adjudication.json`, `pipeline_log.jsonl`, `search_trace.json` | the audit trail |

`--task auto` (the default) lets the wording decide whether to look for a screen
(`perturbation`) or a labeled table (`per_cell`); `--task per_cell` /
`--task perturbation` pin it. This stage only fetches and plans: it never
trains, even if the fetch finds a perfect gold table.

## The review — human, between the stages

Read `data.md`, then fix the plan if the task or the tables are wrong. The
common corrections are:

* the **target / condition column** is not the column the question is about;
* a table the mechanical rules refused (it lived inside an archive, or its
  labels were in the file beside it) is actually the data;
* the gold is metadata, and the measurements sit in an `unusable` sibling.

Edit `plan.json` in place, or override on the stage-2 command line
(`--condition-column`, `--control-label`, `--task`, ...). Nothing else in the
directory needs to change, and the raw fetched plan is never rewritten by the
experiment stage.

## Stage 2 — experiment (never fetches)

```bash
.venv/bin/python -m kosmos.cli.main experiment \
  "Does knocking out the gene STAT1 with CRISPR change the transcriptome of human macrophages?" \
  --data-plan artifacts/dataprep/stat1/plan.json \
  --task perturbation --max-epochs 50 --yes
```

Trains the reviewed plan and writes `run/summary.md`, `run/metrics.json` and
`run/figures/` **beside the plan**. There is no retrieval, no hypothesis
generation and no code generation: if the plan cannot be trained, it fails here
with the reason rather than quietly going back to the network. A finished run is
never overwritten — a re-run lands in `run-2`, `run-3`, ... — so a review-and-
retrain loop keeps every attempt.

## Connecting the two stages again

The stages are a de-coupling, not a fork: the plan is the same JSON contract
`kosmos run --data-plan` already reads, and `plan.json` is what
`scripts/auto_task_run.py` (the fetcher) writes. So a reviewed package can still
be driven through the full autonomous loop:

```bash
.venv/bin/python -m kosmos.cli.main run "<question>" \
  --data-plan artifacts/dataprep/stat1/plan.json --yes
```

and the fully automatic path (no plan: fetch and train in one call) is
unchanged.

## Library seam

The stages are thin wrappers over the same functions the loop uses
(`kosmos/ppi/stages.py`):

* `fetch_stage(...)` → `fetch_and_plan(...)` (writes `plan.json` + reports),
* `experiment_stage(...)` → `run_data_task(..., plan_path=..., fetch=False)`.

That is why the split needs no second implementation of retrieval or training:
only the entry points differ.
