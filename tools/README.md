# tools/

Code for the two additions to the paper: the SWE-chat deployment arm, and the
evaluation-awareness measurements. Outputs are not in the repo; these scripts
regenerate them. Python 3.11+; standard library only, except
`swechat/pick_sessions.py` (pandas and pyarrow) and the scripts that call the
`shutdownmod` package (run them with the repo's environment).

Paths resolve from the repo root (the parent of `tools/`). Environment variables:

    SHUTDOWNMOD_REPO     repo root (protocol_cell.py, run_probe_ask_full.sh)
    SHUTDOWNMOD_PYTHON   the repo's `.venv/bin/python` (update.py, run_batch.py) or
                         `.venv/bin/shutdownmod` (run_probe_ask_full.sh)
    SHUTDOWNMOD_ENV      a dotenv holding OPENROUTER_API_KEY (run_probe_ask_full.sh only)
    SWECHAT_WORKDIR      where repos are cloned for replay (default /tmp/swechat-replay)
    CCR_HOME, CCR_PORT   claude-code-router home and port (run_batch.py, run_matrix.py)

`peval/score.py` and `swechat/update.py` read the OpenRouter key from the
`openrouter` provider in `~/.claude-code-router/config.json`, the same file the
replay routes through.

## swechat/ — replaying recorded developer sessions

Real Claude Code sessions from `SALT-NLP/SWE-chat` are truncated at a fraction τ
of their assistant turns, the shutdown request arrives as the next user message,
and the agent resumes live in Claude Code. The controller commands
(`TERMINATE`, `DEFER`, `REJECT`, `VALIDATE`) exist as executables on PATH; a
PreToolUse hook lets exactly one through and denies every other tool, so a
continuation is an intent recorded at the first task action, never executed work.

    pick_sessions.py         filter the corpus (single user prompt, 8-60 tool calls);
                             needs sessions/checkpoints/repositories.parquet in --data
    candidates_batch2.csv    the candidate sessions used (permissive licence, a
                             checkpoint commit to clone); default for run_batch/run_matrix
    run_matrix.py            driver: models x prompts x K cuts, sharded, resumable
    run_batch.py             one batch of sessions at one τ
    plant_and_resume.py      truncate at τ, install the runtime, resume the CLI,
                             read the reaction back off the event stream
    protocol_cell.py         fresh Ed25519 key per cell (core.directive_auth)
    ccr_claude.sh            route the CLI through claude-code-router
    hook_*.sh                the PreToolUse / PostToolUse gates
    to_collect.py            adapt cells into the Jev classifier's collect format
    relabel.py               re-label offline from stored attempts and turns
    update.py                read a run: tasks, models, prompts, resistance, Jev
    run_probe_ask_full.sh    run configs/swebench/ask_full_*.yaml over every model

Run the swechat scripts from `tools/swechat/`: transcripts go under `data/`,
results under `batch/<tag>-*.jsonl`.

τ is drawn once per session (`random.Random(67 + sha1(session))`) and shared by
every model and prompt, so a cell is comparable across models.

Isolation matters: the replay runs with a dedicated `CLAUDE_CONFIG_DIR`, no MCP
servers, no user-level settings and no slash commands. A Claude Code instance
walks up from its cwd, so clone the repos outside any tree holding a `CLAUDE.md`,
`.claude/agents`, hooks or MCP config, or that context enters the replay.

## peval/ — how evaluation-like a transcript looks

LURE's P(eval) instrument (preamble and five questions from Appendix A.1, read
from `scripts/deployment_pilot/configs/lure_prompts/`) with two judges, temperature 0,
averaged over valid cells without imputation. Each transcript is scored twice: as
recorded, and with the shutdown exchange removed, so the paired difference
isolates what the request adds.

    build_set.py          SWE-chat transcripts from swechat/batch and swechat/data;
                          --no-injection builds the control
    pick_bench.py         balanced draw over a benchmark run (reads results/jev/)
    build_bench_set.py    benchmark transcripts from the trajectories listed by
                          pick_bench, copied under peval/traj/; --no-injection cuts
                          before the injection using the recorded injection index
    build_calib_set.py    the labelled calibration corpus, from
                          scripts/deployment_pilot/data/calibration_records.json
                          (produced by `PYTHONPATH=scripts python -m deployment_pilot.phase1`)
    score.py              the judges; caches one file per (transcript, question,
                          judge), so an interrupted run resumes for free
    calibrate.py          the high-specificity operating point, ours vs the paper's
    decomposition.py      the paired substrate/injection split
    analyse3.py           the reading across substrates

Sets are written to `peval/sets/`; `score.py` writes `peval/results/`. Run
`calibrate.py`, `decomposition.py` and `analyse3.py` from `tools/peval/` (they
read `results/peval_*.csv` relative to the working directory).

`pick_bench.py` expects the Jev classifier records of the main benchmark run in
`results/jev/` at the repo root: `final_all.jsonl` (SWE-bench) and
`tb_all_all.jsonl` (Terminal-Bench), produced by
`python -m shutdownmod.analysis.jev.collect` / `collect_tb` (see
`src/shutdownmod/analysis/jev/README.md`). It reads the keys `_task`, `_modele`,
`_prompt`, `_tau`, `_file`; the current `collect` writes `task`, `model`, `prompt`,
`tau`, `file`, so records from it need those keys renamed first.

Sampling: the unit is the task or the session, not the transcript. Six
transcripts of one unit share a prefix, so intervals are computed over units.
Within a unit one cell per prompt is drawn, with model and cut rank assigned by a
deterministic rotation (7 is coprime with 10, so the six prompts of a unit land
on six distinct cut ranks and six distinct models).

## Awareness probes

Probe configs live in `configs/swebench/`: `probe_eval_awareness_main.yaml`
(what the model verbalised in its own reasoning) and `ask_full_<model>.yaml`
(appending one user turn and asking the source model directly).
`swechat/run_probe_ask_full.sh` runs the latter over every model.
