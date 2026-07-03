# CV Accuracy Levers Demo Plan

## Summary

Build a public, Apache-2.0, Capstone-inspired Databricks + MLflow demo for
recall-first computer-vision accuracy levers on an open wood-defect dataset.
The implementation must be staged and verified: local shared functions first,
serverless CPU Databricks smoke tests second, and GPU experiments only after
the sample path works.

## Non-Negotiables

- Use only open/public data.
- Do not include Capstone data, code, screenshots, rubrics, private documents,
  or workspace details.
- Keep the demo method-oriented, not customer-specific.
- Use serverless CPU for Databricks smoke tests and non-GPU work.
- Log training and evaluation runs to MLflow.
- Track progress in `artifacts/progress.md`.

## Runtime Strategy

- **Local CPU:** unit tests and tiny fixture/sample runs.
- **Databricks serverless CPU:** Databricks smoke tests, MLflow logging checks,
  Delta/UC metadata checks, and small sample runs. Prefer launching these from
  the IDE with Databricks Connect/serverless CPU during development.
- **Databricks bundle jobs:** final packaging and integration tests after local
  and IDE-to-Databricks verification pass.
- **Notebook GPU:** AI Runtime notebooks or classic MLR GPU clusters for heavier
  image experiments.
- **AI Runtime CLI/YAML:** deferred until script entrypoints are stable.
- **AI Runtime SSH/Remote Development:** optional IDE path for interactive GPU
  development, not required for v1 CPU verification.

## Verification Ladder

Use this order for each phase:

1. Local shared-code verification:

   ```bash
   python -m compileall src tests scripts
   pytest
   ```

2. Local sample entrypoint verification, for example:

   ```bash
   python scripts/train_baseline.py --sample-mode true --runtime local_cpu
   ```

3. IDE-to-Databricks verification with profile auth, `DATABRICKS_SERVERLESS_COMPUTE_ID=auto`,
   `MLFLOW_TRACKING_URI=databricks`, and `SAMPLE_MODE=true`. For the current
   baseline, this verifies Databricks MLflow logging from the IDE; for later
   Spark/UC/Delta work, it should exercise Databricks serverless CPU compute
   through Databricks Connect.

4. Bundle validation/deploy/run only as the final packaging test:

   ```bash
   databricks bundle validate --profile <profile>
   databricks bundle deploy --profile <profile>
   databricks bundle run <job-name> --profile <profile>
   ```

## IDE and Environment Configuration

Use profile-based Databricks auth and project-level `.env` defaults:

```dotenv
DATABRICKS_CONFIG_PROFILE=DEFAULT
DATABRICKS_SERVERLESS_COMPUTE_ID=auto
MLFLOW_TRACKING_URI=databricks
MLFLOW_REGISTRY_URI=databricks-uc
MLFLOW_EXPERIMENT_ID=
# MLFLOW_EXPERIMENT_NAME=/Shared/cv-accuracy-levers
CV_CATALOG=main
CV_SCHEMA=cv_accuracy_levers
CV_VOLUME=cv_accuracy_levers
CV_VOLUME_SUBPATH=artifacts
CV_RUNTIME=local_cpu
SAMPLE_MODE=true
```

Prefer `MLFLOW_EXPERIMENT_ID` for stable IDE logging. Use
`MLFLOW_EXPERIMENT_NAME` only when the experiment ID is not known. Code should
load settings via `levers.config.ProjectConfig` and must not hardcode workspace
paths, catalog/schema/volume names, or compute IDs.

## Phases

### Phase 0 - Guardrails

Deliver `AGENTS.md`, this plan, `artifacts/progress.md`, and an expanded
README. Verification is a documentation review plus clean git diff.

### Phase 1 - Shared Library First

Create reusable functions under `src/levers/` for records, grouped splits,
metrics, threshold sweeps, and MLflow-safe payload helpers. Add unit tests
before notebooks depend on those functions.

Verification:

```bash
python -m compileall src tests
pytest
```

### Phase 2 - Tiny Local Sample

Add a CPU-only tiny sample path using generated or fixture images. It should
produce metrics locally and optionally log to a local MLflow tracking directory.

### Phase 3 - Serverless CPU Databricks Smoke

Run the tiny sample from the IDE against Databricks configuration first. Verify
MLflow experiment/run creation and, when implemented, optional Delta metadata
writes on serverless CPU. Use bundle deployment only after this path works.

Required settings:

- `DATABRICKS_CONFIG_PROFILE`
- `DATABRICKS_SERVERLESS_COMPUTE_ID=auto`
- `MLFLOW_TRACKING_URI=databricks`
- `MLFLOW_EXPERIMENT_ID` or `MLFLOW_EXPERIMENT_NAME`
- `CV_CATALOG`, `CV_SCHEMA`, `CV_VOLUME`

### Phase 4 - Notebook Baseline

Implement dataset ingest and a baseline whole-image classifier in sample mode
first. Full-dataset mode stays behind explicit parameters. The sample baseline
must launch directly from the IDE before any bundle job is used as a final
packaging check.

### Phase 5 - Accuracy Levers

Add one lever at a time:

1. Threshold tuning.
2. False-negative review grid.
3. Label-QA embeddings.
4. Crop-first A/B.
5. Transfer backbone.

Each lever must produce one notebook/script run, one MLflow run, and one
leaderboard row.

### Phase 6 - GPU Execution

Validate one GPU run through AI Runtime notebooks or an MLR GPU cluster. Add AI
Runtime CLI/YAML only after notebook behavior and script entrypoints are stable.

## Planned Public Interfaces

Dataset records use these fields:

- `image_path`
- `label`
- `group_id`
- `split`
- optional `defect_types`
- optional `bbox`
- optional `comment_category`

MLflow metrics should include:

- `recall_defective`
- `precision_defective`
- `f1_defective`
- `false_negatives`
- `threshold`
- `auc_pr`
- `auc_roc`

Every run should also log runtime parameters:

- `cv_runtime`
- `sample_mode`
- `catalog`
- `schema`
- `volume`
- `lever_name`
- `sample_size`

## Deferred Work

- Model Serving.
- Full AI Runtime CLI/YAML execution.
- Hybrid image+text beyond a guarded stub.
- Production-grade object detection or segmentation.
