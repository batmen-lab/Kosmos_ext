"""When the rules and the model disagree about a table.

Two judgements are made about every fetched table: the mechanical one (is the
target column present, are the features numeric, does the schema match) and the
model's, from the head of the file. Usually they agree. When they do not, the
direction of the disagreement decides what happens:

| mechanical | model   | who decides                                   |
|------------|---------|-----------------------------------------------|
| usable     | usable  | agreement                                     |
| not usable | not     | agreement -- unusable                         |
| usable     | not     | **the model vetoes**: demote, never promote   |
| not usable | usable  | **a person**, because the model is claiming    |
|            |         | the rules misread the file                    |

The last row is the only case where the model argues for *more* access than the
rules allow, and it is also the case where it is sometimes right -- a headerless
file whose columns are values looks unusable to a rule and obvious to a reader.
Neither side can settle that, so neither does: the case is parked with its
evidence and answered by a person, or left parked and reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Resolved = Literal["gold", "supplementary", "unusable", "pending"]

USABLE_ROLES = {"gold", "supplementary", "unlabeled", "bad_label"}


@dataclass
class Disagreement:
    """One table the rules and the model read differently."""

    path: str
    mechanical_role: str
    mechanical_reason: str
    model_role: str
    model_reason: str
    raw_head: list[str] = field(default_factory=list)
    #: The model's one concrete manual step that would make the table usable,
    #: when it named one. A person deciding the case should see it.
    model_salvage: str = ""

    @property
    def needs_human(self) -> bool:
        return self.mechanical_role not in USABLE_ROLES and self.model_role in USABLE_ROLES

    def question(self) -> str:
        """The text a person answers: the evidence, then the decision."""
        lines = [
            "The rules and the model disagree about this table:",
            f"  {self.path}",
            "",
            "Its first lines:",
            *[f"    {line[:160]}" for line in self.raw_head[:5]],
            "",
            f"  rules say: {self.mechanical_role} — {self.mechanical_reason[:200]}",
            f"  model says: {self.model_role} — {self.model_reason[:200]}",
            *(
                [f"  a person could rescue it: {self.model_salvage[:200]}"]
                if self.model_salvage
                else []
            ),
            "",
            "Decide: [g]old / [e]vidence (labels ignored) / [r]eject / [s]kip for now",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "mechanical": {"role": self.mechanical_role, "reason": self.mechanical_reason},
            "model": {"role": self.model_role, "reason": self.model_reason},
            "raw_head": self.raw_head,
            "needs_human": self.needs_human,
        }


ANSWER_ROLES = {
    "g": "gold",
    "gold": "gold",
    "e": "supplementary",
    "evidence": "supplementary",
    "supplementary": "supplementary",
    "r": "unusable",
    "reject": "unusable",
    "unusable": "unusable",
    "s": "pending",
    "skip": "pending",
}


def resolve(mechanical_role: str, model_role: str | None) -> Resolved:
    """Apply the table above, without asking anyone."""
    if model_role is None:
        return _normalise(mechanical_role)
    mechanical_ok = mechanical_role in USABLE_ROLES
    model_ok = model_role in USABLE_ROLES
    if mechanical_ok and model_ok:
        # Both accept it; the model may still say "not gold, only evidence".
        return _normalise("supplementary" if model_role != "gold" else "gold")
    if not mechanical_ok and not model_ok:
        return "unusable"
    if mechanical_ok and not model_ok:
        return "unusable" if model_role == "unusable" else "supplementary"
    return "pending"  # rules refuse, the model accepts: a person decides


def _normalise(role: str) -> Resolved:
    if role in {"gold", "supplementary", "unusable", "pending"}:
        return role  # type: ignore[return-value]
    if role in {"unlabeled", "bad_label"}:
        return "supplementary"
    return "unusable"


def parse_answer(answer: str) -> Resolved:
    """Map a person's reply to a role; anything unrecognised means 'leave it'."""
    return ANSWER_ROLES.get((answer or "").strip().lower(), "pending")
