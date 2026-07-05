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
CV_DATA_SOURCE=manifest
CV_DATA_MANIFEST=
CV_DATA_DIR=
CV_UC_TABLE=image_manifest
CV_RUNTIME=local_cpu
SAMPLE_MODE=true
```

Prefer `MLFLOW_EXPERIMENT_ID` for stable IDE logging. Use
`MLFLOW_EXPERIMENT_NAME` only when the experiment ID is not known. Code should
load settings via `levers.config.ProjectConfig` and must not hardcode workspace
paths, catalog/schema/volume names, compute IDs, or dataset locations.

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

Implement a baseline whole-image classifier in sample mode first. Full-dataset
mode stays behind explicit parameters. The sample baseline must launch directly
from the IDE before any bundle job is used as a final packaging check.

### Phase 4.5 - Public Dataset Ingest

Add a license-first manifest ingest path before real-data training and Phase 5
levers. The default source is `manifest`; optional RF100-VL or other external
source adapters must remain env-driven and must not commit images, archives, or
credentials. Reject non-commercial and unreviewed custom licenses by default.
The ingest script can optionally copy referenced images into the configured UC
volume, write the normalized manifest under that volume, and register the
manifest as a Delta table in `CV_CATALOG.CV_SCHEMA`.

### Phase 5 - Accuracy Levers

Add one lever at a time:

1. Threshold tuning.
2. False-negative review grid.
3. Label-QA embeddings.
4. Crop-first A/B.
5. Transfer backbone.

Each lever must produce one notebook/script run, one MLflow run, and one
leaderboard row.

#### Phase 5.1 - Threshold Tuning

Use the existing sample baseline prediction scores to measure threshold tuning
as an operating-point lever. Do not add a new classifier, full-dataset training,
or GPU execution in this phase.

Public interface:

```bash
python scripts/tune_threshold.py --sample-mode true --runtime local_cpu
python scripts/tune_threshold.py --sample-mode true --runtime local_cpu --log-mlflow --tracking-uri file:/tmp/cv-accuracy-levers-threshold-mlruns
```

The run must log `lever_name=threshold_tuning`, sample/runtime/catalog/schema/
volume params, min recall, split/feature seeds, baseline threshold, tuned test
metrics, validation metrics, fixed-0.5 metrics, tuned-minus-fixed deltas, and a
threshold sweep artifact. The comparison claim is limited to changing recall
and precision on the same split, not improving the underlying model.

#### Phase 5.2 - False-Negative Review Grid

Use the existing sample baseline prediction scores and selected threshold to
produce a ranked false-negative review artifact. Do not add embeddings, crop
logic, visual model changes, full-dataset training, or GPU execution in this
phase.

Public interface:

```bash
python scripts/review_false_negatives.py --sample-mode true --runtime local_cpu
python scripts/review_false_negatives.py --sample-mode true --runtime local_cpu --review-threshold 0.95 --log-mlflow --tracking-uri file:/tmp/cv-accuracy-levers-error-review-mlruns
```

The default review threshold is the validation-selected threshold. The sample
data may have no false negatives at that threshold; use `--review-threshold`
only to exercise non-empty review artifacts in deterministic smoke tests. The
run must log `lever_name=false_negative_review`, sample/runtime/catalog/schema/
volume params, min recall, split/feature seeds, selected threshold, review
threshold, review metrics, ranked false-negative rows, a review summary, and a
leaderboard row artifact.

#### Phase 5.3 - Label-QA Embeddings

Use deterministic synthetic feature embeddings from the existing sample
baseline path to produce label-quality review artifacts. This phase is
analysis-only: do not add CLIP, Torch, crop-first logic, full-dataset image
embedding extraction, new classifier training, or GPU execution.

Public interface:

```bash
python scripts/review_label_quality_embeddings.py --sample-mode true --runtime local_cpu
python scripts/review_label_quality_embeddings.py --sample-mode true --runtime local_cpu --inject-synthetic-label-issue --log-mlflow --tracking-uri file:/tmp/cv-accuracy-levers-label-quality-mlruns
```

The run must log `lever_name=label_quality_embeddings`, sample/runtime/catalog/
schema/volume params, min recall, split/feature seeds, selected threshold,
`embedding_source=sample_baseline_synthetic_features_v1`, similarity thresholds,
neighbor count, review top-k, synthetic issue injection flag, review metrics,
`label_quality_review_rows.json`, `label_quality_summary.json`,
`embedding_neighbors.json`, `leaderboard_row.json`, and `predictions.json`.
Treat all rows as review candidates only; synthetic injection exists only to
exercise non-empty artifact paths and must not be presented as a real dataset
defect.

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

Normalized ingest rows must also retain:

- `source`
- `source_license`
- `original_label`

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
