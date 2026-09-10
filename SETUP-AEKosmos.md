# Kosmos_ext (branch `mark-AEKosmos`) — environment notes

What was done to bring this fork to parity with the working Kosmos at
`Omics/Kosmos`, and the few things that are not obvious from the code.

## Environment

```bash
.venv/bin/kosmos run "<question>" --data-path <csv> -o report.md --no-cache
```

- venv: `.venv`, Python 3.11.9. `pip check` reports no broken requirements.
- Kosmos installed editable against THIS fork; AutoEvidence editable against
  `2026_yanglu_omics_os/src/AutoEvidence`, with the `[mcp,hf]` extras.
- `.env` holds the OpenRouter config, model `deepseek/deepseek-v4-pro-0813`.
  It is gitignored — do not commit it, this fork has a public remote.

**AutoEvidence's `mcp` and `hf` are optional extras**, and both imports are
function-scope. `pip install -e AutoEvidence` alone therefore produces an
environment that imports fine and fails at the first gateway call. Install as
`pip install -e '<path>/AutoEvidence[mcp,hf]'`.

## This directory's name contains a space — that matters

`Kosmos Batmen Lab` puts a space in every absolute path here, and command
strings (`server:` lines, `KOSMOS_FINDER_SERVER`) are split with `shlex.split`
before being spawned. An unquoted path splits at the space and the consumer
tries to execute `/Omics/Kosmos` — a DIRECTORY — failing with
`PermissionError: [Errno 13]` on a path that really exists, so it reads like a
permissions problem rather than a quoting one.

Quote the executable in the finder export:

```bash
export KOSMOS_FINDER_SERVER="'<abs>/Kosmos_ext/.venv/bin/autoevidence-serve' --finder-config '<abs>/AutoEvidence/finder.yaml' --key '<abs>/AutoEvidence/keys/capsule_signing.key'"
```

`kosmos/datasearch/emit.py` had the same bug when WRITING configs: it quoted
every argument but interpolated the serve binary raw. Fixed here (and in
`Omics/Kosmos`, to keep the two identical) — quoted only when it names an
existing file, so a multi-token command like `python -m autoevidence.server`
still works.

## Running several runs at once

Each run writes hypotheses and experiments to `kosmos.db`. Concurrent runs on
one database let one run's hypotheses trip the NOVELTY FILTER against another's
— the second run degrades silently rather than failing. Give each its own:

```bash
DATABASE_URL="sqlite:///t1.db" .venv/bin/kosmos run ...
```

## Test configs in `data/`

- `t1_single_hf/` — one public HuggingFace dataset, no policy.
- `t2_multi_fibrosis/` — four HF parquets, each behind a policy in
  `policies/`. Those policies pin `accepted_snapshots`; verify with
  `.venv/bin/autoevidence snapshot --source '<ref>'` if a run is refused.
- `t3_find_data/` — no config committed on purpose: pointing
  `--evidence-config` at a path that does NOT exist makes the run search for a
  dataset and write the config itself.

## Test suite

```bash
DATABASE_URL="sqlite:///probe.db" .venv/bin/python -m pytest tests/unit/execution tests/unit/cli -c /dev/null -q
```

Expect **20 failures** — the identical set the reference Kosmos produces (18
`test_commands.py` CLI tests, 2 Docker health checks). `-c /dev/null` bypasses
`pytest.ini`'s coverage flags.
