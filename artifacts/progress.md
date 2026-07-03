# Progress

Update this file whenever a phase starts, completes, is blocked, or is
verified. Keep entries short and include exact verification commands.

## Phase Checklist

| Phase | Status | Verification | Notes |
|---|---|---|---|
| 0 - Guardrails | Complete | Documentation created | Added repo plan, AGENTS.md, README updates. |
| 1 - Shared library first | Complete | `python -m compileall src tests scripts`; `pytest` | Added unit-tested split/metrics/threshold helpers. |
| 2 - Tiny local sample | Complete | `PYTHONPATH=src python scripts/run_tiny_sample.py`; `PYTHONPATH=src python scripts/run_tiny_sample.py --log-mlflow --tracking-uri file:/tmp/cv-accuracy-levers-mlruns` | CPU-only deterministic sample path prints metrics and logs to local MLflow. |
| 3 - Serverless CPU Databricks smoke | Complete | `python -m compileall src tests scripts`; `pytest`; `python scripts/run_tiny_sample.py --runtime databricks_serverless_cpu`; `DATABRICKS_AUTH_STORAGE=plaintext databricks bundle validate --profile fevm`; `DATABRICKS_AUTH_STORAGE=plaintext databricks bundle deploy --profile fevm`; `DATABRICKS_AUTH_STORAGE=plaintext databricks bundle run smoke_sample_cpu --profile fevm` | Serverless CPU smoke job completed successfully and logged MLflow metrics. |
| 4 - Notebook baseline | Local complete | `python -m compileall src tests scripts`; `pytest`; `python scripts/train_baseline.py --sample-mode true --runtime local_cpu`; `python scripts/train_baseline.py --sample-mode true --runtime local_cpu --log-mlflow --tracking-uri file:/tmp/cv-accuracy-levers-baseline-mlruns` | Sample-mode baseline launches locally without bundle deployment. Serverless CPU bundle job is added but not required for local verification. |
| 5 - Accuracy levers | Not started | Pending | One lever at a time. |
| 6 - GPU execution | Not started | Pending | AI Runtime notebook or MLR GPU first. |

## Log

- 2026-07-02: Initialized staged plan, verification rules, and progress
  tracking.
- 2026-07-02: Added shared library, notebook placeholders, Databricks bundle
  skeleton, and tiny sample script. Verified compile, unit tests, and local tiny
  sample run.
- 2026-07-02: Verified local MLflow logging to
  `file:/tmp/cv-accuracy-levers-mlruns`. `databricks bundle validate` is blocked
  by expired Databricks CLI auth: run `databricks auth login` or configure the
  intended profile before Databricks smoke verification.
- 2026-07-02: Added `.env.example`, `ProjectConfig`, MLflow/UC runtime
  parameter guidance, and serverless CPU Databricks Connect environment
  conventions for the next implementation session. Verified with
  `python -m compileall src tests scripts`, `pytest`,
  `PYTHONPATH=src python scripts/run_tiny_sample.py`, and
  `PYTHONPATH=src python scripts/run_tiny_sample.py --log-mlflow --tracking-uri file:/tmp/cv-accuracy-levers-mlruns`.
- 2026-07-02: Resumed Phase 3 serverless CPU smoke work. `databricks bundle
  validate` is still blocked by Databricks CLI auth refresh; hardening the smoke
  script so it can run without caller-provided `PYTHONPATH`.
- 2026-07-02: Hardened Phase 3 smoke path: `scripts/run_tiny_sample.py` now
  resolves local `src/` itself, the Databricks job explicitly passes
  `--log-mlflow`, and `cv_runtime` is logged from the actual runtime argument.
  Verified with `python -m compileall src tests scripts`, `pytest`, and
  `python scripts/run_tiny_sample.py --runtime databricks_serverless_cpu`.
  `databricks bundle validate` remains blocked by Databricks CLI auth: run
  `databricks auth login` for the intended profile, then re-run bundle
  validation/deploy/run.
- 2026-07-02: Resuming Phase 3 after `.env` update to use the intended
  Databricks profile. Running bundle validation with `.env` loaded.
- 2026-07-02: Refreshed Databricks CLI auth for `fevm`. Added a serverless job
  environment for the smoke Python task after validation reported that
  serverless tasks require an environment. Verified `DATABRICKS_AUTH_STORAGE=plaintext databricks bundle validate --profile fevm`.
- 2026-07-02: Deployed the bundle and started the Phase 3 smoke job on
  Databricks serverless CPU. Run failed because Databricks executed the Python
  file without defining `__file__`; updating the script path bootstrap for that
  execution mode before redeploying.
- 2026-07-02: Redeployed after `__file__` fallback fix. Serverless smoke run
  reached script completion but failed because Databricks treated
  `SystemExit(0)` as a task exception; updating script exit handling before
  rerunning.
- 2026-07-03: Completed Phase 3. Verified local compile/tests, local
  Databricks-runtime sample mode, bundle validation, bundle deployment, and
  serverless CPU smoke run with MLflow logging through the `fevm` profile.
- 2026-07-03: Started Phase 4 baseline work. Adding tested sample-mode baseline
  helpers, `scripts/train_baseline.py`, a thin baseline notebook wrapper, a
  serverless CPU bundle job, and deferred AI Runtime CLI example files.
- 2026-07-03: Locally verified Phase 4 baseline. `python -m compileall src tests scripts`
  passed, `pytest` passed with 19 tests, `python scripts/train_baseline.py --sample-mode true --runtime local_cpu`
  passed, and local MLflow logging passed with
  `python scripts/train_baseline.py --sample-mode true --runtime local_cpu --log-mlflow --tracking-uri file:/tmp/cv-accuracy-levers-baseline-mlruns`.
  Databricks bundle deployment is not required for local launch; keep
  `baseline_sample_cpu` as an optional later serverless CPU smoke path.
- 2026-07-03: Corrected the deferred AIR example to use the documented
  `air run --file` workflow and AIR config shape. Verified YAML syntax with
  `python -c "import yaml; yaml.safe_load(open('air/baseline_sample.yaml')); print('yaml_ok')"`.
  AIR CLI validation was not run because `air` is not installed on the local
  PATH.
- 2026-07-03: Started documentation update for Phase 4/5 verification order:
  prefer local IDE launch and IDE-to-Databricks compute smoke checks before
  bundle validation/deploy/run. Bundle execution should be the final packaging
  test, not the default development verification step.
- 2026-07-03: Completed verification-order documentation update in `AGENTS.md`,
  `README.md`, and `artifacts/cv-accuracy-levers-demo-plan.md`. Verified the
  wording with
  `rg -n "IDE-to-Databricks|bundle validation|final packaging|DATABRICKS_SERVERLESS_COMPUTE_ID=auto|Verification Ladder|databricks bundle" AGENTS.md README.md artifacts/cv-accuracy-levers-demo-plan.md artifacts/progress.md`
  and confirmed code still compiles with `python -m compileall src tests scripts`.
