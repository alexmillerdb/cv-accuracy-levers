# cv-accuracy-levers

Open-source computer-vision accuracy levers demo for recall-first binary image
classification on an open wood-defect dataset. The project is Databricks +
MLflow oriented and Capstone-inspired, but it must not contain customer data,
customer code, customer screenshots, private rubrics, or workspace details.

## Goal

Show a staged, measurable workflow for improving defective-wood recall without
pretending that one model change fixes every accuracy problem:

1. Start with tested shared functions and a tiny CPU sample.
2. Validate Databricks connectivity and MLflow logging on serverless CPU.
3. Add one accuracy lever at a time and record each result in MLflow.
4. Move to GPU only after the CPU/sample path is working.

The first-class metrics are defective-class recall, precision, false negatives,
F1, PR AUC, ROC AUC, and the selected operating threshold.

## Runtime Paths

- **Local CPU:** unit tests and tiny fixture/sample runs.
- **Databricks serverless CPU:** smoke tests, Delta/UC metadata checks, and
  small MLflow-logged runs.
- **Notebook GPU:** AI Runtime notebooks or classic MLR GPU clusters for
  heavier image experiments.
- **AI Runtime CLI:** deferred until script entrypoints are stable; intended for
  reproducible remote GPU jobs rather than the first implementation pass.

## IDE Configuration

Copy `.env.example` to `.env` for local IDE defaults. Keep secrets out of
`.env`; use Databricks CLI/OAuth profile auth instead.

```dotenv
DATABRICKS_CONFIG_PROFILE=DEFAULT
DATABRICKS_SERVERLESS_COMPUTE_ID=auto
MLFLOW_TRACKING_URI=databricks
MLFLOW_REGISTRY_URI=databricks-uc
MLFLOW_EXPERIMENT_ID=
CV_CATALOG=main
CV_SCHEMA=cv_accuracy_levers
CV_VOLUME=cv_accuracy_levers
CV_VOLUME_SUBPATH=artifacts
CV_DATA_SOURCE=manifest
CV_DATA_MANIFEST=/path/to/public_manifest.jsonl
CV_DATA_DIR=/path/to/public_images
CV_UC_TABLE=image_manifest
```

Prefer `MLFLOW_EXPERIMENT_ID` for IDE-to-Databricks logging. Use
`MLFLOW_EXPERIMENT_NAME` only when the ID is not known.

## Planned Flow

The notebook and script flow is intentionally phased:

1. `00_setup_and_ingest.py` - open dataset ingest and binary label mapping.
2. `01_baseline_resnet.py` - whole-image baseline.
3. `02_threshold_tuning.py` - recall/precision operating point sweep.
4. `03_error_review_gut_check.py` - high-confidence false-negative review.
5. `04_label_quality_embeddings.py` - suspected label issues and leakage checks.
6. `05_crop_first_ab_test.py` - crop-first A/B test.
7. `06_transfer_backbone.py` - frozen feature backbone comparison.
8. `07_recommendation_summary.py` - MLflow leaderboard and next action table.
9. `99_future_hybrid_fusion.py` - guarded future stub for categorical comments.

## Local Verification

```bash
python -m compileall src tests scripts
pytest
```

These commands must pass before notebook work depends on shared functions.

The deterministic tiny sample can also log to local MLflow:

```bash
python scripts/run_tiny_sample.py \
  --log-mlflow \
  --tracking-uri file:/tmp/cv-accuracy-levers-mlruns
```

After Databricks auth is configured, the same script can be run against
Databricks MLflow by setting `.env` and using `MLFLOW_TRACKING_URI=databricks`.

## IDE-to-Databricks Verification

Use this order while developing:

1. Local compile/tests and sample script launch.
2. IDE-launched Databricks smoke checks with profile auth and serverless CPU.
3. Bundle validation/deploy/run only as the final packaging test.

For IDE-to-Databricks settings, start from `.env.example` and set:

```dotenv
DATABRICKS_CONFIG_PROFILE=<profile>
DATABRICKS_SERVERLESS_COMPUTE_ID=auto
MLFLOW_TRACKING_URI=databricks
MLFLOW_REGISTRY_URI=databricks-uc
MLFLOW_EXPERIMENT_ID=<experiment-id>
CV_RUNTIME=databricks_serverless_cpu
SAMPLE_MODE=true
```

For the current Phase 4 baseline, the model computation is local CPU but the
run can verify IDE-to-Databricks auth and MLflow logging without deploying a
job bundle:

```bash
python scripts/train_baseline.py \
  --sample-mode true \
  --runtime databricks_serverless_cpu \
  --log-mlflow \
  --tracking-uri databricks
```

When a phase adds Spark, UC, or Delta operations, use the same IDE environment
with `DATABRICKS_SERVERLESS_COMPUTE_ID=auto` to run the Databricks Connect code
from the IDE before packaging it as a bundle job.

Reserve these commands for the final packaging/integration test:

```bash
databricks bundle validate --profile <profile>
databricks bundle deploy --profile <profile>
databricks bundle run baseline_sample_cpu --profile <profile>
```

## Public Dataset Ingest

Phase 4.5 adds a manifest-backed ingest path before real-data training. The
default source is `manifest`, which keeps downloads, credentials, and image
artifacts outside git while normalizing records into the project schema.

Input manifests can be CSV, JSON, or JSONL and must include:

- `image_path`
- `label` or `class_name` or `category`
- `group_id`
- `source`
- `source_license`

The ingest path rejects non-commercial and unreviewed custom licenses by
default. Use permissively licensed public data for the default Apache-2.0 demo
path. Wood-specific non-commercial datasets can be documented as research-only
external inputs, but should not become the default workflow.

Run a local sample ingest:

```bash
python scripts/prepare_dataset.py \
  --source manifest \
  --manifest-path /path/to/public_manifest.jsonl \
  --data-dir /path/to/public_images \
  --output-path artifacts/ingest/normalized_manifest.jsonl \
  --sample-mode true \
  --sample-size 100 \
  --runtime local_cpu
```

`artifacts/ingest/` and `data/` are ignored so generated manifests and images
do not get staged accidentally. The optional `rf100vl` source is guarded by
`ROBOFLOW_API_KEY`; direct download is deferred until the source and dataset
selection are explicitly reviewed.

To persist the normalized manifest to Unity Catalog and stage images under the
configured UC volume, run from Databricks or an IDE session with Databricks
Connect/serverless CPU configured:

```bash
python scripts/prepare_dataset.py \
  --source manifest \
  --manifest-path /path/to/public_manifest.jsonl \
  --data-dir /path/to/public_images \
  --sample-mode true \
  --runtime databricks_serverless_cpu \
  --catalog "$CV_CATALOG" \
  --schema "$CV_SCHEMA" \
  --volume "$CV_VOLUME" \
  --volume-subpath "$CV_VOLUME_SUBPATH" \
  --copy-images-to-uc-volume \
  --uc-image-upload-mode sdk \
  --databricks-profile "$DATABRICKS_CONFIG_PROFILE" \
  --write-uc \
  --uc-table image_manifest \
  --create-uc-objects
```

`--create-uc-objects` attempts to create `CV_CATALOG`, `CV_SCHEMA`, and
`CV_VOLUME`; omit it when those objects already exist or your user lacks create
privileges. The UC table is written to `CV_CATALOG.CV_SCHEMA.<uc-table>`, and
the normalized manifest is also written under `CV_VOLUME_SUBPATH/ingest/` in
the configured volume.

Use `--uc-image-upload-mode sdk` from an IDE or local terminal. Use the default
`local` mode only when the script runs inside Databricks where `/Volumes/...`
is mounted.

The bundle also includes `ingest_manifest_uc_cpu` as a final packaged
serverless CPU path. Set the `data_manifest`, `data_dir`, and `uc_table` bundle
variables before running it; the manifest and image directory must be
Databricks-accessible.

## Baseline Verification

Phase 4 adds a sample-mode whole-image baseline entrypoint:

```bash
python scripts/train_baseline.py --sample-mode true --runtime local_cpu
```

This launches locally and does not require Databricks bundle validation,
deployment, or a Databricks job. The bundle job is only for a later remote
serverless CPU smoke run.

The baseline uses public-safe synthetic whole-image feature vectors until open
dataset ingest is implemented. It trains a small centroid scorer, chooses a
recall-first threshold on the validation split, evaluates on the test split, and
prints/logs the same metrics that later levers must use.

To log the baseline to a local MLflow directory:

```bash
python scripts/train_baseline.py \
  --sample-mode true \
  --runtime local_cpu \
  --log-mlflow \
  --tracking-uri file:/tmp/cv-accuracy-levers-baseline-mlruns
```

For final packaged Databricks serverless CPU validation, use the bundle job
after local and IDE-to-Databricks checks pass:

```bash
databricks bundle run baseline_sample_cpu --profile <profile>
```

Do not compare levers against the baseline unless both runs use the same split
and threshold-selection logic.

## Threshold Tuning Verification

Phase 5.1 adds a sample-mode threshold tuning lever over the existing baseline
prediction scores:

```bash
python scripts/tune_threshold.py --sample-mode true --runtime local_cpu
```

This run selects a validation threshold that satisfies the configured recall
floor, evaluates that selected threshold on the test split, and compares it
against a fixed `0.5` operating point on the same test predictions. Treat the
result as a recall/precision operating-point comparison, not as evidence that
the underlying classifier improved.

To log the threshold tuning run to a local MLflow directory:

```bash
python scripts/tune_threshold.py \
  --sample-mode true \
  --runtime local_cpu \
  --log-mlflow \
  --tracking-uri file:/tmp/cv-accuracy-levers-threshold-mlruns
```

The MLflow run logs tuned test metrics, validation metrics, fixed-0.5 test
metrics, tuned-minus-fixed deltas, `threshold_sweep.json`,
`threshold_tuning_summary.json`, and the baseline prediction rows.

For IDE-to-Databricks MLflow smoke verification:

```bash
python scripts/tune_threshold.py \
  --sample-mode true \
  --runtime databricks_serverless_cpu \
  --log-mlflow \
  --tracking-uri databricks
```

For final packaged Databricks serverless CPU validation, use the bundle job
after local and IDE-to-Databricks checks pass:

```bash
databricks bundle run threshold_tuning_sample_cpu --profile <profile>
```

## AI Runtime CLI

AI Runtime CLI support is deferred infrastructure for reproducible GPU
submissions. The example contract in [`air/baseline_sample.yaml`](air/baseline_sample.yaml)
calls `scripts/train_baseline.py` in `SAMPLE_MODE=true` with
`CV_RUNTIME=ai_runtime_cli`. Validate that YAML against the installed `air` CLI
schema before submitting it; AIR is not required for local or serverless CPU
verification.

## Progress

Implementation progress is tracked in
[`artifacts/progress.md`](artifacts/progress.md). Future agents should update
that file whenever a phase is started, completed, blocked, or verified.
