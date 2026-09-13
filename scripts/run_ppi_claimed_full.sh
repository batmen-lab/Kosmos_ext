#!/usr/bin/env bash
# One-command full claimed flow: official Kosmos PPI run + persisted analysis
# + generated Markdown report.
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export WORLD_MODEL_ENABLED=false
export REDIS_ENABLED=false
export ENABLE_SANDBOXING=false
export OPENAI_TIMEOUT=60
export USE_LITERATURE_CONTEXT=false
export REQUIRE_NOVELTY_CHECK=false
export NUM_HYPOTHESES=1

# PPI configuration (linear model + in-sample pseudo labels, seed 42)
export PPI_MODEL_DESIGN=linear
export PPI_PSEUDO_MODE=in_sample
export PPI_SEED=42
export PPI_STAGE1_EPOCHS=4
export PPI_STAGE2_EPOCHS=1
export PPI_EXTERNAL_PER_DONOR=5000
export PPI_MAX_EXTERNAL_SAMPLES=20000
export PPI_MAX_EPOCHS=20
export PPI_PATIENCE=5

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/kosmos-mpl
# Local venv (stable Python 3.11 + PyTorch); override with KOSMOS_PYTHON=...
PYTHON="${KOSMOS_PYTHON:-$ROOT/.venv/bin/python}"

START=$(date -u +"%Y-%m-%d %H:%M:%S")

QUESTION="Does adding pseudo-labeled external non-gold RNA data (PPILoss correction) improve cross-donor cell-type classification versus gold-only training, with a linear model and in-sample pseudo-label mode? Train donor 13272 (80/20 dev), evaluate donor 19593, use external_unlabeled as unlabeled PPI evidence, and report baseline vs PPI accuracy, balanced accuracy, macro F1, per-class recall changes, and an evidence-based analysis of which cell types benefit or degrade."

echo "=== [1/3] Official Kosmos research cycle (hypothesis -> design -> PPI -> analysis) ==="
"$PYTHON" -m kosmos.cli.main \
  run "$QUESTION" \
  --domain biology \
  --max-iterations 1 \
  --budget 5 \
  --data-path "$ROOT/data/csv_donor_splits/gold_13272_19593_labeled.csv" \
  --external-data-path "$ROOT/data/csv_donor_splits/external_unlabeled.csv" \
  --output "$ROOT/artifacts/ppi_claimed_full_run.json"

echo "=== [2/3] Persisting/reading analysis from database ==="
"$PYTHON" - "$START" <<'PY'
import json
import sqlite3
import sys
from datetime import datetime

start = sys.argv[1]
con = sqlite3.connect("kosmos.db")
con.row_factory = sqlite3.Row
row = con.execute(
    "select id, data, interpretation, supports_hypothesis, created_at "
    "from results where created_at >= ? order by created_at desc limit 1",
    (start,),
).fetchone()
if row is None:
    raise SystemExit("No new result found; research did not complete.")

d = json.loads(row["data"])
interp = json.loads(row["interpretation"]) if row["interpretation"] else {}
exp = con.execute(
    "select hypothesis_id, description from experiments where id = "
    "(select experiment_id from results where id = ?)",
    (row["id"],),
).fetchone()
hyp = None
if exp and exp["hypothesis_id"]:
    hyp = con.execute(
        "select statement, rationale from hypotheses where id = ?",
        (exp["hypothesis_id"],),
    ).fetchone()

ft = d["final_test_metrics"]
rows = "| metric | baseline | PPI | delta |\n|---|---|---|---|\n"
for m in ("accuracy", "balanced_accuracy", "macro_f1"):
    b, p = ft["baseline"][m], ft["ppi"][m]
    rows += f"| {m} | {b:.4f} | {p:.4f} | {p-b:+.4f} |\n"

analysis_text = interp.get("summary") or interp.get("significance_interpretation") or ""
findings = interp.get("key_findings") or []
findings_md = "\n".join(f"- {f}" for f in findings[:10]) or "- (none)"

md = f"""# Kosmos full PPI research cycle (claimed workflow)

Result DB id: `{row['id']}` ({row['created_at']})

## Hypothesis
{hyp['statement'] if hyp else "N/A"}

## Final test (donor 19593, baseline vs PPI)
{rows}

## DataAnalyst analysis
{analysis_text}

### Key findings
{findings_md}

Supports hypothesis: {row['supports_hypothesis']}
"""
out = "artifacts/ppi_claimed_full_report.md"
open(out, "w", encoding="utf-8").write(md)
print(f"Analysis written to {out}")
print("--- analysis preview ---")
print(analysis_text[:2000])
PY

echo "=== [3/3] Full report ==="
echo "artifacts/ppi_claimed_full_report.md"
