# Repository Instructions

## Purpose

This is a public, Apache-2.0, Capstone-inspired computer-vision accuracy levers
demo. It must stay public-safe: do not add Capstone data, customer code,
customer screenshots, partner rubrics, workspace URLs, private transcripts, or
other proprietary artifacts.

## Working Style

- Work in phases. Do not implement multiple major phases at once unless the
  current phase verification already passes.
- Update `artifacts/progress.md` when starting, completing, blocking, or
  verifying a phase.
- Prefer small, reviewable changes with concrete verification over large
  unverified scaffolds.
- Put reusable logic in `src/levers/` before notebooks depend on it.
- Add or update unit tests for shared functions before notebook integration.
- Keep notebooks as thin orchestration layers around tested functions.

## Verification Rules

- For local/shared-code changes, run:

  ```bash
  python -m compileall src tests
  pytest
  ```

- For script/notebook phases, prefer direct IDE launches before bundle work:

  ```bash
  python scripts/train_baseline.py --sample-mode true --runtime local_cpu
  ```

- For Databricks smoke tests, use serverless CPU unless GPU is explicitly
  required.
- For Databricks compute verification during development, prefer IDE-to-Databricks
  execution with `DATABRICKS_SERVERLESS_COMPUTE_ID=auto` over deploying the
  entire bundle. Use this path for Spark/UC/Delta or serverless CPU checks that
  need Databricks compute but do not need job packaging.
- Treat `databricks bundle validate`, `databricks bundle deploy`, and
  `databricks bundle run ...` as final packaging/integration tests after the
  local and IDE-to-Databricks smoke paths pass. Do not make bundle deployment the
  default verification gate for ordinary script or notebook changes.
- Every Databricks run should record whether it used local CPU, serverless CPU,
  MLR GPU, AI Runtime notebook, AI Runtime SSH, or AI Runtime CLI.
- Every training run should log parameters and metrics to MLflow.
- Use `SAMPLE_MODE=true` or an equivalent tiny sample setting before running the
  full dataset.
- Do not claim a lever improves recall until it has an apples-to-apples MLflow
  comparison against the baseline on the same split.

## Environment and MLflow Configuration

- Use `.env` for project defaults and keep secrets in Databricks CLI auth,
  OAuth, or workspace auth tooling. Never commit `.env`.
- Start from `.env.example` when configuring a new IDE session.
- Use `DATABRICKS_CONFIG_PROFILE` to select the Databricks profile from
  `~/.databrickscfg`.
- Use `DATABRICKS_SERVERLESS_COMPUTE_ID=auto` for Databricks Connect
  serverless CPU smoke tests. Use `DATABRICKS_CLUSTER_ID` only as an explicit
  classic-compute fallback.
- Use `MLFLOW_TRACKING_URI=databricks` for IDE-to-Databricks tracking.
- Use `MLFLOW_REGISTRY_URI=databricks-uc` when model registry behavior is
  needed.
- Prefer `MLFLOW_EXPERIMENT_ID` over `MLFLOW_EXPERIMENT_NAME` so code does not
  hardcode `/Users/...` or `/Shared/...` workspace paths.
- Parameterize Unity Catalog locations with `CV_CATALOG`, `CV_SCHEMA`,
  `CV_VOLUME`, and `CV_VOLUME_SUBPATH`.
- Shared code should read these settings through `levers.config.ProjectConfig`
  rather than reading many environment variables directly.
- Every MLflow run should log runtime context including `CV_RUNTIME`,
  `SAMPLE_MODE`, catalog/schema/volume, sample size, and the lever name.

## Runtime Guidance

- Local CPU is for unit tests and tiny sample runs.
- IDE-launched Databricks serverless CPU is the default for compute-facing smoke
  tests, UC/Delta metadata checks, and non-GPU Databricks execution.
- Bundle-deployed Databricks jobs are the last verification step for packaging,
  job wiring, and release readiness.
- Notebook code should remain compatible with serverless CPU sample mode and
  GPU full mode where practical.
- AI Runtime notebooks or MLR GPU clusters are the first GPU execution targets.
- AI Runtime CLI/YAML should be added only after script entrypoints are stable.
- AI Runtime SSH/Remote Development is optional IDE infrastructure for
  interactive GPU work; it is not required for CPU verification.

## Modeling Scope

- Optimize for defective-class recall with explicit precision tradeoffs.
- Track false negatives, precision, F1, PR AUC, ROC AUC, and threshold.
- Use group-aware splits; never use per-image random splits when multiple views
  of a source item may exist.
- Treat label noise and leakage as first-class risks.
- Hybrid image+text is future/stub work only unless leakage and
  inference-availability checks are implemented.
- Do not add AGPL, non-commercial, gated, or custom-license model dependencies
  without an explicit licensing review.

## Subagents

Use subagents for read-heavy research, independent test-log triage, and
parallel review. Avoid parallel write-heavy work unless write scopes are
explicitly disjoint.
