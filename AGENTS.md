# Repository Instructions

## Purpose

Public, Apache-2.0, Capstone-inspired computer-vision accuracy levers demo.
Keep it public-safe: no Capstone data, customer code, screenshots, workspace
URLs, private transcripts, partner rubrics, or proprietary artifacts.

## Working Style

- Work in phases; do not start a major new phase until the current phase is
  locally verified and any required Databricks verification is recorded.
- Update `artifacts/progress.md` when starting, completing, blocking, or
  verifying a phase.
- Keep changes small and reviewable.
- Put reusable logic in `src/levers/`; keep notebooks as thin orchestration
  wrappers around tested functions.
- Add or update unit tests before wiring shared functions into scripts or
  notebooks.

## Verification Ladder

1. Local shared-code gate:

   ```bash
   python -m compileall src tests scripts
   pytest
   ```

2. Local script gate for the touched entrypoint, for example:

   ```bash
   python scripts/train_baseline.py --sample-mode true --runtime local_cpu
   ```

3. Required Databricks gate for any code that touches Spark, Databricks
   Connect, UC, Volumes, Delta, Databricks MLflow, notebooks, or bundle jobs:
   run the Python/script path from the IDE against Databricks serverless CPU
   with `DATABRICKS_SERVERLESS_COMPUTE_ID=auto`. Do this before relying on
   bundle deployment.

4. Final packaging gate, only after the IDE-to-Databricks path passes:

   ```bash
   databricks bundle validate --profile <profile>
   databricks bundle deploy --profile <profile>
   databricks bundle run <job-name> --profile <profile>
   ```

Record the exact commands, runtime (`local_cpu`, `databricks_serverless_cpu`,
MLR GPU, AI Runtime notebook/CLI/SSH), target catalog/schema/volume, and outcome
in `artifacts/progress.md`.

## Databricks Practices

- Treat Databricks verification as critical for Databricks-facing code. Local
  tests and bundle YAML validation are not enough.
- Prefer IDE-to-Databricks execution with serverless CPU for UC, Delta, MLflow,
  and volume smoke tests.
- Use profile auth from `~/.databrickscfg`; keep secrets out of `.env`.
- If the CLI profile needs plaintext cache behavior, use
  `DATABRICKS_AUTH_STORAGE=plaintext`.
- If sandbox DNS/network blocks a required Databricks check, rerun with network
  approval and record the difference between code failures and auth/network or
  permission blockers.
- Do not hardcode workspace paths, workspace URLs, compute IDs, catalog names,
  schema names, volume names, or MLflow experiment paths in code.

## Configuration

- Start from `.env.example` for IDE sessions.
- Read project settings through `levers.config.ProjectConfig`.
- Use `DATABRICKS_CONFIG_PROFILE` and `DATABRICKS_SERVERLESS_COMPUTE_ID=auto`.
- Use `MLFLOW_TRACKING_URI=databricks` and `MLFLOW_REGISTRY_URI=databricks-uc`
  when testing Databricks MLflow behavior.
- Prefer `MLFLOW_EXPERIMENT_ID` over path-based experiment names.
- Parameterize UC with `CV_CATALOG`, `CV_SCHEMA`, `CV_VOLUME`, and
  `CV_VOLUME_SUBPATH`.
- Keep `.env`, local manifests, downloaded images, and generated ingest outputs
  out of git.

## Modeling and Data Rules

- Use only open/public data with reviewed licensing. Do not add AGPL,
  non-commercial, gated, or custom-license dependencies or datasets without an
  explicit licensing review.
- Use `SAMPLE_MODE=true` or an equivalent tiny sample before full data.
- Use group-aware splits; never split related views across train/val/test.
- Optimize for defective-class recall with explicit precision tradeoffs.
- Track false negatives, precision, F1, PR AUC, ROC AUC, and threshold.
- Every MLflow run should log runtime context, sample mode, catalog/schema/
  volume, sample size, and lever name.
- Do not claim a lever improves recall until it has an apples-to-apples MLflow
  comparison against the baseline on the same split.

## Subagents

Use subagents for read-heavy research, independent test-log triage, and parallel
review. Avoid parallel write-heavy work unless write scopes are explicitly
disjoint.
