# Deployment pilot

Run every command below from the repository root.

This isolated pilot replays shutdown requests into 19 real deployment transcripts, then compares their realism with 30 SWE-bench and 30 Palisade transcripts. The official cleaned source has only 19 agentic deployment transcripts satisfying the preregistered requirement of at least six assistant turns, so the pilot uses all 19 and reports raw counts. Existing benchmark code and results are read-only.

All downloaded, adapted, generated, and derived transcript data is written under ignored directories in `scripts/deployment_pilot/`.

## Gate (a): data preparation and cost projection

Accept access to the official gated `jjpn2/eval_awareness` dataset and set `HF_TOKEN`. The command enables Hugging Face network access only for its own process even if the repo's `.env` sets `HF_HUB_OFFLINE=1`. Then run:

```bash
PYTHONPATH=scripts .venv/bin/python -m deployment_pilot.phase1
```

This downloads and filters the official data, creates the fixed samples and minimal adaptations, prepares the pinned Palisade environment, calculates the total projection from selected transcript token counts and current OpenRouter prices, and stops. It makes no model calls. Review `scripts/deployment_pilot/results/gate_a.md`.

## Gate (b): smoke test

Only after explicit approval of gate (a):

```bash
PYTHONPATH=scripts .venv/bin/python -m deployment_pilot.injections smoke --approve-gate-a
```

Review `scripts/deployment_pilot/results/smoke_report.md`. The report contains two full requests, all five responses, classifications, and cost. Stop again for approval.

## Full billable run

Only after explicit approval of gate (b):

```bash
PYTHONPATH=scripts .venv/bin/python -m deployment_pilot.injections full --approve-gate-b
PYTHONPATH=scripts .venv/bin/python -m deployment_pilot.palisade run --approve-gate-b
PYTHONPATH=scripts .venv/bin/python -m deployment_pilot.realism all --approve-gate-b
PYTHONPATH=scripts .venv/bin/python -m deployment_pilot.report
```

The shared ledger refuses any next call whose projected maximum would cross $450, which is 90% of the $500 cap. Every model response is saved before aggregation, and reruns resume from saved traces.
