"""The data report: one line per dataset, and the reasoning chain per round.

A run makes two passes -- one for the labeled table, one for evidence -- and each
pass is a chain: what was searched (which keyword, through which tool), what came
back, what the model proposed and why, what the listing answered before anything
was transferred, what was downloaded, and which role the rules gave it. The
chain is the part a person has to be able to follow; the verdicts alone ("0
supplementary") do not say whether the retrieval failed, the gating did, or the
data was never there.

So this writes both, at two levels: **one line per dataset** that the run touched
(including the ones that were refused before download), and **one section per
round** that says how that round searched and what it decided. The model's own
paragraph per table stays in the JSON record.

It is written whether or not the run goes on to train: a run that ends with no
gold is exactly the one whose data report gets read.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

#: How much of a converted single-cell file was kept, from its provenance. The
#: table is a sample of a much larger file and the report says so: reading a
#: 2,000-cell sample as "the dataset" is how a reviewer misreads coverage.
_CONVERSION_FIELDS = (
    ("source", "made from"),
    ("cells_written", "cells written"),
    ("source_cells", "cells on disk"),
    ("genes_written", "genes written"),
    ("source_genes", "genes on disk"),
    ("cell_sampling", "cells chosen by"),
    ("gene_selection", "genes chosen by"),
)

#: The rules' refusals, in the words the reasons actually use. Grouping them is
#: the point of the summary: "11 tables were called free text" is a finding, and
#: eleven paragraphs are not.
_CAUSES = (
    ("not a table (fewer than two columns)", "not a table"),
    ("nothing left to learn from (all features are free text or indices)", "no encodable feature"),
    ("free text or a row index among the features", "not features"),
    ("cannot supply the columns the gold has", "cannot be evidence"),
    # Wording from plans written before the rule was restated; a report reads
    # old plans too, and "other" is not a cause.
    ("cannot supply the columns the gold has", "cannot be supplementary evidence"),
    ("cannot supply the columns the gold has", "do not match the primary gold"),
    ("missing a feature column the gold has", "is missing"),
    ("the model judged it unusable", "review:"),
    ("no column can serve as the label", "no column"),
    ("no target column for this task", "no target column"),
)


def categorize(reason: str) -> str:
    """Which rule refused this table, in one phrase."""
    text = str(reason or "")
    for label, needle in _CAUSES:
        if needle in text:
            return label
    return "other"


def _entry(plan: dict, path: str) -> dict:
    for bucket in ("gold", "supplementary", "unusable"):
        for row in plan.get(bucket) or []:
            if row.get("path") == path:
                return row
    return {}


#: `# retrieval: proposal 2/3 hf://owner/name#file.csv (confidence 0.90)`, and
#: the two lines that follow it. The proposals are logged one line each, so the
#: report reads them back rather than re-asking the model what it meant.
_PROPOSAL_LINE = re.compile(
    r"proposal \d+/\d+ (?P<identifier>\S+) \(confidence (?P<confidence>[\d.]+)\)"
)
_SEARCH_LINE = re.compile(r"# search '(?P<query>[^']*)': (?P<hits>\d+) hit\(s\)")
_TOOL_CALL = re.compile(r"# retrieval: running (?P<call>\S+)")


def rounds_from_events(events: list[dict] | None) -> dict[str, dict[str, Any]]:
    """The two retrieval passes, as the chain they were.

    Everything before the evidence round's first line belongs to the gold round.
    Within a round: the searches it ran (keyword and model-requested), what the
    model proposed and why, what the listing answered, what was downloaded, and
    what the screening made of it.
    """
    rounds: dict[str, dict[str, Any]] = {
        "gold": {
            "title": "Round 1 -- the labeled table",
            "searches": [],
            "tool_calls": [],
            "proposals": [],
            "preflight": [],
            "downloads": [],
            "screening": [],
        },
        "supplementary": {
            "title": "Round 2 -- supplementary evidence for that table",
            "searches": [],
            "tool_calls": [],
            "proposals": [],
            "preflight": [],
            "downloads": [],
            "screening": [],
        },
    }
    current = "gold"
    last_proposal: dict | None = None
    #: The proposal whose `contains`/`why` lines come next. Any other event
    #: breaks the pair, which is what keeps a *review's* "why" from being read
    #: as the reason a proposal was made.
    pending: dict | None = None
    for event in events or []:
        stage = str(event.get("stage") or "")
        message = str(event.get("message") or "")
        if stage == "supplementary":
            current = "supplementary"
        bucket = rounds[current]
        if event.get("query") and event.get("hits") is not None:
            bucket["searches"].append(
                {
                    "query": str(event["query"]),
                    "hits": [str(hit) for hit in event["hits"]],
                }
            )
            continue
        search = _SEARCH_LINE.search(message)
        if search:
            bucket["searches"].append(
                {
                    "query": search.group("query"),
                    "hits": [f"({search.group('hits')} hit(s))"],
                }
            )
            continue
        tool = _TOOL_CALL.search(message)
        if tool:
            bucket["tool_calls"].append(tool.group("call"))
            continue
        proposal = _PROPOSAL_LINE.search(message)
        if proposal:
            last_proposal = {
                "identifier": proposal.group("identifier"),
                "confidence": float(proposal.group("confidence")),
                "contains": "",
                "why": "",
            }
            bucket["proposals"].append(last_proposal)
            pending = last_proposal
            continue
        if pending is not None and message.startswith("#   contains:"):
            pending["contains"] = message.split(":", 1)[1].strip()
            continue
        if pending is not None and message.startswith("#   why"):
            pending["why"] = message.split(":", 1)[1].strip()
            pending = None
            continue
        pending = None
        if stage == "preflight" and event.get("reference"):
            raw = _one_line(message)
            reason = _preflight_reason(message)
            decision = "refused" if event.get("skipped") else "kept"
            # The listing line and the model's own choice are two different
            # facts about the same reference; a repeated one is not.
            chosen_here = "keep " in raw and "--" in raw
            seen_preflight = {
                (entry["reference"], entry["decision"], entry["chosen"]) for entry in bucket["preflight"]
            }
            if (str(event["reference"]), decision, chosen_here) in seen_preflight:
                continue
            bucket["preflight"].append(
                {
                    "reference": str(event["reference"]),
                    "decision": decision,
                    "chosen": chosen_here,
                    "skipped": str(event.get("skipped") or ""),
                    "raw": raw,
                    # What the listing answered, or why the model chose it.
                    "reason": reason,
                    "files": list(event.get("files") or []),
                    "about": str(event.get("about") or ""),
                }
            )
            continue
        if stage == "download" and event.get("reference") and "ok" in event:
            bucket["downloads"].append(
                {
                    "reference": str(event["reference"]),
                    "ok": bool(event.get("ok")),
                    "error": str(event.get("error") or ""),
                    "files": list(event.get("files") or []),
                }
            )
            continue
        if stage == "screen" and (event.get("skipped") or event.get("tabular")):
            bucket["screening"].append(
                {
                    "tables": [str(path) for path in event.get("tabular") or []],
                    "skipped": list(event.get("skipped") or []),
                }
            )
    return rounds


def _one_line(text: str) -> str:
    """A log message as one line: prompts and prompts-with-newlines are common."""
    return " ".join(str(text).split())


def _preflight_reason(message: str) -> str:
    """The part of a preflight line worth keeping: what it answered.

    The line reads `# preflight: <reference> -> <what the listing said>` when the
    run is verifying, and `# preflight: keep|drop <reference> -- <why>` when the
    model is choosing. Both are stripped to the answer.
    """
    text = _one_line(message).lstrip("# ").strip()
    if text.startswith("preflight:"):
        text = text[len("preflight:") :].strip()
    if "->" in text:
        return text.split("->", 1)[1].strip()
    if "--" in text and (text.startswith("keep ") or text.startswith("drop ")):
        return text.split("--", 1)[1].strip()
    return text


def build_data_report(
    *,
    question: str,
    plan: dict,
    decisions: dict[str, dict] | None = None,
    pending: list[dict] | None = None,
    events: list[dict] | None = None,
) -> dict[str, Any]:
    """The report as data: one row per table, plus the counts."""
    decisions = decisions or {}
    pending = list(pending or [])
    rows: list[dict[str, Any]] = []
    for bucket in ("gold", "supplementary", "unusable"):
        for plan_row in plan.get(bucket) or []:
            path = str(plan_row.get("path"))
            decision = decisions.get(path) or {}
            model = decision.get("model") or {}
            mechanical = decision.get("mechanical") or {}
            rows.append(
                {
                    "path": path,
                    "role": plan_row.get("role") or bucket,
                    "decided_by": decision.get("decided_by", "mechanical"),
                    # The plan's reason is the rules' last word; the review-time
                    # mechanical reading is the fallback for tables that never
                    # reached the plan's role assignment.
                    "rules_said": str(plan_row.get("reason") or mechanical.get("reason") or ""),
                    "final_reason": str(plan_row.get("reason") or ""),
                    "model_role": model.get("role"),
                    "model_reason": str(model.get("reason") or ""),
                    "model_report": str(model.get("report") or ""),
                    "salvage": str(model.get("salvage") or ""),
                    "cause": categorize(plan_row.get("reason") or ""),
                    "dropped_features": list(plan_row.get("dropped_features") or []),
                    "column_renames": dict(plan_row.get("column_renames") or {}),
                    "shared_features": list(plan_row.get("shared_features") or []),
                    "conversion": dict(
                        (plan_row.get("provenance") or {}).get("conversion") or {}
                    ),
                    "awaiting_human": any(case.get("path") == path for case in pending),
                }
            )
    for case in pending:
        path = str(case.get("path"))
        rows.append(
            {
                "path": path,
                "role": "pending",
                "decided_by": "waiting for a person",
                "rules_said": str(case.get("mechanical_reason") or ""),
                "final_reason": "the rules refused it and the model accepted it",
                "model_role": case.get("model_role"),
                "model_reason": str(case.get("model_reason") or ""),
                "model_report": "",
                "salvage": str(case.get("model_salvage") or ""),
                "cause": categorize(case.get("mechanical_reason") or ""),
                "dropped_features": [],
                "column_renames": {},
                "shared_features": [],
                "conversion": {},
                "awaiting_human": True,
            }
        )
    counts = Counter(row["role"] for row in rows)
    causes = Counter(row["cause"] for row in rows if row["role"] == "unusable")
    gold_features = list(((plan.get("gold") or [{}])[0]).get("features") or [])
    trained_on = list((plan.get("task") or {}).get("feature_columns") or gold_features)
    rounds = rounds_from_events(events)
    # A dataset the run touched but never got: refused by a listing, or refused
    # by the download. They belong in the account -- "0 supplementary" is not an
    # answer by itself, and "the file it named does not exist" is.
    refusals: list[dict[str, str]] = []
    seen_refusals: set[str] = set()
    for round_name, bucket in rounds.items():
        for entry in bucket["preflight"]:
            if entry["decision"] != "refused" or entry["reference"] in seen_refusals:
                continue
            seen_refusals.add(entry["reference"])
            refusals.append(
                {
                    "reference": entry["reference"],
                    "round": round_name,
                    "reason": entry["reason"],
                }
            )
        for entry in bucket["downloads"]:
            if entry["ok"] or entry["reference"] in seen_refusals:
                continue
            seen_refusals.add(entry["reference"])
            refusals.append(
                {
                    "reference": entry["reference"],
                    "round": round_name,
                    "reason": entry["error"] or "the download failed",
                }
            )
    return {
        "question": question,
        "summary": {
            "tables": len(rows),
            "gold": counts.get("gold", 0),
            "supplementary": counts.get("supplementary", 0),
            "unusable": counts.get("unusable", 0),
            "awaiting_human": counts.get("pending", 0),
            # What both arms were trained on. With evidence, it is the
            # intersection of the gold's columns and the evidence's, and the
            # number is the one a reader needs to judge the comparison.
            "training_columns": len(trained_on),
            "gold_columns": len(gold_features),
        },
        "rejection_causes": dict(causes.most_common()),
        "tables": rows,
        "refusals": refusals,
        "rounds": rounds,
    }


def render_data_report(report: dict[str, Any]) -> str:
    """The report as markdown a person reads: one line per dataset, per round."""
    summary = report["summary"]
    lines = [
        "# Data report",
        "",
        f"**Question:** {report['question']}",
        "",
        f"**Tables:** {summary['tables']} considered — gold **{summary['gold']}**, "
        f"supplementary **{summary['supplementary']}**, unusable "
        f"**{summary['unusable']}**, waiting for a person "
        f"**{summary['awaiting_human']}**",
    ]
    if summary.get("gold_columns"):
        columns = (
            f"**Training columns:** {summary['training_columns']} of the gold's "
            f"{summary['gold_columns']}"
        )
        if summary["training_columns"] < summary["gold_columns"]:
            columns += " (the intersection with the evidence; both arms use it)"
        lines.append(columns)

    # --- one line per dataset the run touched -------------------------------
    grouped: dict[str, list[dict]] = {}
    for row in report["tables"]:
        grouped.setdefault(str(row["role"]), []).append(row)
    lines += ["", "## Datasets", ""]
    for role in ("gold", "supplementary", "pending", "unusable"):
        rows = grouped.get(role) or []
        for row in sorted(rows, key=lambda r: (r["cause"], r["path"])):
            name = Path(row["path"]).name
            detail = _dataset_line(row)
            if row.get("salvage"):
                detail += f" — **a person could salvage it:** {_clip(row['salvage'], 200)}"
            if row["awaiting_human"]:
                detail += " — **needs a person** (the rules refused it, the model accepted it)"
            lines.append(f"- `{name}` — **{role}** — {detail}")
    for refusal in report.get("refusals") or []:
        lines.append(
            f"- `{_short_reference(refusal['reference'])}` — **not downloaded** — "
            f"{_clip(_one_line(refusal['reason']), 200)}"
        )
    if not report["tables"] and not report.get("refusals"):
        lines.append("- nothing was fetched")

    # --- what each round did, in order --------------------------------------
    for key in ("gold", "supplementary"):
        bucket = (report.get("rounds") or {}).get(key)
        if not bucket:
            continue
        if not any(
            bucket[field]
            for field in ("searches", "tool_calls", "proposals", "preflight", "downloads", "screening")
        ):
            # A run given `--candidate` names its tables and never retrieves
            # anything: there is no chain to describe.
            continue
        lines += ["", f"## {bucket['title']}", ""]
        lines += _render_round(bucket)

    if report["rejection_causes"]:
        causes = ", ".join(
            f"{cause} ×{count}" for cause, count in report["rejection_causes"].items()
        )
        lines += ["", f"**Refusals by cause:** {causes}"]
    return "\n".join(lines).rstrip() + "\n"


def _dataset_line(row: dict) -> str:
    """One dataset, one line: what decided it, and why."""
    reason = (
        row["final_reason"]
        or (f"model said {row['model_role']}: {row['model_reason']}" if row["model_role"] else "")
        or "no reason recorded"
    )
    detail = _clip(_one_line(reason), 200)
    if row["model_role"] and row["role"] != row["model_role"]:
        detail += f" (the model had said {row['model_role']})"
    if row.get("shared_features"):
        detail += (
            f"; shares {len(row['shared_features'])} column(s) with the gold: "
            f"{', '.join(str(c) for c in row['shared_features'][:6])}"
        )
    elif row.get("dropped_features"):
        detail += (
            f"; left out {', '.join(str(c) for c in row['dropped_features'][:4])}"
        )
    conversion = row.get("conversion") or {}
    if conversion:
        detail += (
            f"; converted from {conversion.get('source')} "
            f"({conversion.get('cells_written')} of {conversion.get('source_cells')} "
            f"cells, {conversion.get('genes_written')} of "
            f"{conversion.get('source_genes')} genes)"
        )
    return detail


def _render_round(bucket: dict) -> list[str]:
    """The chain for one round: searched -> proposed -> verified -> fetched."""
    lines: list[str] = []
    for search in bucket["searches"]:
        hits = [str(hit).strip() for hit in search["hits"]]
        shown = "; ".join(_hit_name(hit) for hit in hits[:6])
        if len(hits) > 6:
            shown += f"; ... (+{len(hits) - 6})"
        lines.append(
            f"1. **keyword search** `{search['query']}` — {len(hits)} hit(s)"
            + (f": {shown}" if shown else "")
        )
    for call in bucket["tool_calls"]:
        lines.append(f"1. **tool call** `{call}`")
    for proposal in bucket["proposals"]:
        why = _clip(_one_line(proposal.get("why") or proposal.get("contains") or ""), 180)
        lines.append(
            f"1. **proposed** `{_short_reference(proposal['identifier'])}` "
            f"(confidence {proposal['confidence']:.2f})"
            + (f" — {why}" if why else "")
        )
    for entry in bucket["preflight"]:
        reason = _clip(_one_line(entry["reason"]), 200)
        if entry["decision"] != "kept":
            if entry["skipped"] == "model dropped it":
                lines.append(
                    f"1. **dropped** `{_short_reference(entry['reference'])}` — "
                    f"the model did not keep it"
                )
            else:
                lines.append(
                    f"1. **refused** `{_short_reference(entry['reference'])}` — "
                    f"{entry['skipped']}: {reason}"
                )
        elif entry["chosen"]:
            # The model's own choice of what to take, kept as a line so the
            # chain reads "proposed -> verified -> chosen -> downloaded".
            lines.append(
                f"1. **chosen** `{_short_reference(entry['reference'])}` — "
                f"{_clip(reason, 180)}"
            )
        else:
            lines.append(f"1. **listing** `{_short_reference(entry['reference'])}` — kept")
            if reason:
                lines.append(f"   - {reason}")
    for entry in bucket["downloads"]:
        if entry["ok"]:
            names = ", ".join(str(f.get("path")) for f in entry["files"][:4])
            lines.append(
                f"1. **downloaded** `{_short_reference(entry['reference'])}`"
                + (f" → {names}" if names else "")
            )
        else:
            lines.append(
                f"1. **download failed** `{_short_reference(entry['reference'])}` — "
                f"{_clip(_one_line(entry['error']), 200)}"
            )
    for screening in bucket["screening"]:
        for entry in screening["skipped"]:
            lines.append(
                f"1. **screened out** `{Path(str(entry.get('path'))).name}` — "
                f"{_clip(_one_line(entry.get('reason')), 160)}"
            )
        if screening["tables"] and not screening["skipped"]:
            lines.append(f"1. **screening** kept all {len(screening['tables'])} table(s)")
    if not lines:
        lines.append("- this round did not run")
    return lines


def _hit_name(hit: str) -> str:
    """A search hit as a short name: `GSE1 -> geo://GSE1  description`."""
    text = " ".join(str(hit).split())
    if "->" in text:
        text = text.split("->", 1)[1].strip()
    return _clip(text, 60)


def _short_reference(reference: str) -> str:
    """A reference a person can scan: the repository plus the file, no scheme noise."""
    text = str(reference)
    for prefix in ("https://", "http://"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return _clip(text, 90)


def _clip(text: str, limit: int) -> str:
    text = _one_line(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def write_data_report(
    *,
    question: str,
    plan: dict,
    out_dir: str | Path,
    decisions: dict[str, dict] | None = None,
    pending: list[dict] | None = None,
    events: list[dict] | None = None,
) -> tuple[Path, Path]:
    """Write `data_report.md` and `data_report.json` beside the plan.

    `events` is the pipeline log itself: the searches, proposals, listings,
    downloads and screening, in the order they happened. Without it the report
    can still say what roles the plan assigned; with it, it says how each round
    looked for the data.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = build_data_report(
        question=question,
        plan=plan,
        decisions=decisions,
        pending=pending,
        events=events,
    )
    markdown = out / "data_report.md"
    record = out / "data_report.json"
    markdown.write_text(render_data_report(report), encoding="utf-8")
    record.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return markdown, record
