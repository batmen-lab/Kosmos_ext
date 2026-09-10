"""Finding data when a run was given a question and nothing to run it on.

Nothing in this package searches anything. The search happens inside an
AutoEvidence discovery server, in an interpreter that is not this one, and what
comes back here is a signed capsule of POINTERS -- accessions and, where
AutoEvidence has a connector that could route one, a reference such as
`hf://owner/name#train`. This package's whole job is to turn one of those
pointers into an `evidence.yaml` a human can read before running it.

The seam is deliberate. A pointer is a third party's unverified claim; a source
is something a steward admitted. `evidence.yaml` is where the second happens,
and it is a file somebody opens, not a value this code passes along.
"""

from __future__ import annotations
