"""Which labeled table should be the primary training set.

Several tables can carry the label column. Picking the biggest one is a rule,
not a judgement, and it is the wrong judgement for this setting: a large cohort
from the wrong population is worth less than a smaller one that actually matches
the question, and the extras still enter as unlabeled evidence either way.

So the choice goes to the model, constrained to the tables that were actually
offered, with the facts it needs (rows, features, where each came from) rather
than a path list. When no model is available the deterministic rule is used and
recorded as such -- a run must not stop because a provider is down, but it must
also not pretend a rule was a judgement.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class GoldChoice:
    path: str
    reason: str
    source: str = "llm"  # llm | fallback


def _facts(entry: dict[str, Any]) -> str:
    bits = [f"rows={entry.get('rows')}", f"features={entry.get('n_features')}"]
    provenance = entry.get("provenance") or {}
    if provenance.get("reference"):
        bits.append(f"source={provenance['reference']}")
    if entry.get("features_sample"):
        bits.append("columns=" + ", ".join(entry["features_sample"][:4]))
    return "; ".join(str(b) for b in bits)


def choose_primary_gold(
    objective: str,
    labeled: Sequence[dict[str, Any]],
    *,
    client: Any = None,
    log: Callable[[str], None] = print,
) -> GoldChoice:
    """Pick the labeled table to train on; the rest become evidence.

    `labeled` entries are plan-style dicts with at least `path` and `rows`.
    """
    if not labeled:
        raise ValueError("no labeled table to choose from")
    if len(labeled) == 1:
        only = labeled[0]
        log(f"# primary gold: {only['path']} (the only labeled table)")
        return GoldChoice(path=only["path"], reason="the only labeled table", source="rule")

    # Rows first, then the *narrower* table. Two tables of the same study often
    # have the same rows -- one per cell -- and one of them carries 129,922
    # columns where the other carries 14,088. Picking by path order put the wide
    # one in charge of the whole run: the gold's schema becomes the intersection
    # every other table has to meet, and the encoder has 129,922 columns to fit.
    fallback = max(
        labeled,
        key=lambda entry: (
            entry.get("rows") or 0,
            -(entry.get("n_features") or 0),
            entry["path"],
        ),
    )
    if client is None:
        log(
            f"# primary gold: {fallback['path']} (largest labeled table; no model "
            f"available to judge relevance)"
        )
        return GoldChoice(
            path=fallback["path"],
            reason=f"largest labeled table ({fallback.get('rows')} rows)",
            source="fallback",
        )

    listing = "\n".join(f"- {entry['path']}\n    {_facts(entry)}" for entry in labeled)
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "why": {"type": "string"},
        },
        "required": ["path", "why"],
    }
    prompt = (
        "A scientific analysis needs one primary labeled training set.\n\n"
        f"Research question:\n{objective}\n\n"
        "Every table below carries the label column. Which one should the model be "
        "trained on?\n"
        f"{listing}\n\n"
        "More rows is NOT automatically better. Prefer the table whose population, "
        "measurement and context actually match the question; the others are not "
        "discarded -- they will be used as unlabeled evidence, so their labels are "
        "not lost to the analysis either way.\n\n"
        "Answer as JSON with 'path' (copied exactly from the list) and 'why' (one "
        "sentence)."
    )
    log(f"# primary gold: asking the model to choose among {len(labeled)} labeled table(s)")
    try:
        response = client.generate_structured(
            prompt=prompt, schema=schema, max_tokens=400, temperature=0
        )
    except Exception as e:  # noqa: BLE001 - fall back, but say why
        log(f"# primary gold: model call failed ({e}); using the largest table")
        return GoldChoice(
            path=fallback["path"],
            reason=f"largest labeled table (model call failed: {e})",
            source="fallback",
        )
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            response = None

    chosen_path = str((response or {}).get("path", "")).strip()
    allowed = {entry["path"] for entry in labeled}
    if chosen_path not in allowed:
        # A path the model invented is discarded, not matched to the closest.
        log(
            f"# primary gold: model named {chosen_path!r}, which is not one of the "
            f"labeled tables; using the largest"
        )
        return GoldChoice(
            path=fallback["path"],
            reason=(
                f"largest labeled table (model named {chosen_path!r}, which is not "
                f"among the candidates)"
            ),
            source="fallback",
        )
    reason = str((response or {}).get("why") or "chosen by the model")
    log(f"# primary gold: {chosen_path}")
    log(f"#   why: {reason}")
    return GoldChoice(path=chosen_path, reason=reason, source="llm")
