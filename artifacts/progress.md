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
| 4.5 - Public dataset ingest | Complete | `python -m compileall src tests scripts`; `pytest`; `python scripts/prepare_dataset.py --source manifest --manifest-path /tmp/cv-accuracy-levers-ingest-smoke/manifest.jsonl --data-dir /tmp/cv-accuracy-levers-ingest-smoke/images --output-path /tmp/cv-accuracy-levers-ingest-smoke/normalized_manifest_uc_copy.jsonl --sample-mode true --sample-size 3 --runtime local_cpu --catalog demo_catalog --schema demo_schema --volume demo_volume --volume-subpath cv --copy-images-to-uc-volume --uc-image-dir /tmp/cv-accuracy-levers-ingest-smoke/uc-volume/images`; `python -c "import yaml; yaml.safe_load(open('databricks.yml')); print('databricks_yml_ok')"`; `DATABRICKS_AUTH_STORAGE=plaintext databricks bundle validate --profile fevm` | Added license-first manifest ingest plus optional UC table/volume persistence before Phase 5 levers. |
| 5 - Accuracy levers | In progress | `python -m compileall src tests scripts`; `pytest`; `python scripts/tune_threshold.py --sample-mode true --runtime local_cpu`; `python scripts/review_false_negatives.py --sample-mode true --runtime local_cpu --review-threshold 0.95`; `python scripts/review_label_quality_embeddings.py --sample-mode true --runtime local_cpu --inject-synthetic-label-issue`; `python scripts/run_crop_first_ab.py --sample-mode true --runtime local_cpu`; `python scripts/run_crop_first_ab.py --sample-mode true --runtime databricks_serverless_cpu --log-mlflow --tracking-uri databricks --experiment-name /Shared/cv-accuracy-levers`; `DATABRICKS_AUTH_STORAGE=plaintext databricks bundle validate --profile fevm`; `DATABRICKS_AUTH_STORAGE=plaintext databricks bundle deploy --profile fevm`; `DATABRICKS_AUTH_STORAGE=plaintext databricks bundle run crop_first_ab_sample_cpu --profile fevm` | Phase 5.1 threshold tuning, Phase 5.2 false-negative review, Phase 5.3 label-quality embeddings, and Phase 5.4 crop-first A/B complete; transfer-backbone not started. |
| 6 - GPU execution | In progress | `python -m compileall src tests scripts`; `pytest`; `python scripts/train_gpu_baseline.py --manifest-path /tmp/cv-accuracy-levers-gpu-smoke/manifest.jsonl --data-dir /tmp/cv-accuracy-levers-gpu-smoke --sample-mode true --sample-size 6 --runtime local_cpu --device cpu --image-size 24 --batch-size 2 --epochs 1`; `python -c "import yaml; yaml.safe_load(open('databricks.yml')); yaml.safe_load(open('air/gpu_baseline_sample.yaml')); yaml.safe_load(open('air/baseline_sample.yaml')); print('yaml_ok')"`; `air --version`; `python scripts/train_gpu_baseline.py --manifest-path data/beans/manifest.jsonl --data-dir data/beans/images --sample-mode true --sample-size 32 --runtime local_cpu --device cpu --image-size 64 --batch-size 4 --epochs 1`; `DATABRICKS_AUTH_STORAGE=plaintext databricks fs cp /tmp/cv-accuracy-levers-gpu-real-manifest.jsonl dbfs:/Volumes/serverless_stable_yau46e_catalog/cv_accuracy_levers/cv_accuracy_levers/artifacts/ingest/gpu_baseline_manifest.jsonl --overwrite --profile fevm`; `DATABRICKS_AUTH_STORAGE=plaintext air run --file air/gpu_baseline_sample.yaml --dry-run -p fevm --json --override env_variables.CV_DATA_MANIFEST="/Volumes/serverless_stable_yau46e_catalog/cv_accuracy_levers/cv_accuracy_levers/artifacts/ingest/gpu_baseline_manifest.jsonl" env_variables.CV_DATA_DIR= env_variables.MLFLOW_EXPERIMENT_ID=""` | AIR CLI-first GPU path scaffolded and dry-run validated with `GPU_1xA10`; local CPU Torch smoke passed on generated smoke data and on the 100-row MIT-licensed Beans real-image sample. UC image and manifest staging are complete for AIR; GPU `--watch` is pending explicit submit. Default catalog is `serverless_stable_yau46e_catalog`. |

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
- 2026-07-03: Started Phase 4.5 public dataset ingest foundation. Building a
  license-first manifest path before Phase 5 threshold tuning so later training
  can run on real public data instead of only synthetic sample features.
- 2026-07-03: Completed Phase 4.5 public dataset ingest foundation. Added
  manifest normalization helpers, `scripts/prepare_dataset.py`, a thin
  `00_setup_and_ingest.py` wrapper, dataset env defaults, docs, and tests.
  Verified with `python -m compileall src tests scripts`, `pytest` with 30
  tests passing, and
  `python scripts/prepare_dataset.py --source manifest --manifest-path /tmp/cv-accuracy-levers-ingest-smoke/manifest.jsonl --data-dir /tmp/cv-accuracy-levers-ingest-smoke/images --output-path /tmp/cv-accuracy-levers-ingest-smoke/normalized_manifest.jsonl --sample-mode true --sample-size 3 --runtime local_cpu`.
- 2026-07-03: Started Phase 4.5 UC ingest extension. Adding optional Unity
  Catalog persistence so normalized manifests can be written to
  `CV_CATALOG.CV_SCHEMA` and files can be staged under `CV_VOLUME`.
- 2026-07-03: Completed Phase 4.5 UC ingest extension. Added UC volume path
  helpers, image-copy staging into a volume path, optional Spark persistence to
  a UC Delta table and volume manifest directory, notebook widgets, README
  guidance, and an optional `ingest_manifest_uc_cpu` bundle job. Verified with
  `python -m compileall src tests scripts`, `pytest` with 34 tests passing,
  local image-copy smoke via `scripts/prepare_dataset.py`, and YAML parsing for
  `databricks.yml`. Initial bundle validation was blocked by sandbox DNS, then
  passed with network access using
  `DATABRICKS_AUTH_STORAGE=plaintext databricks bundle validate --profile fevm`.
  Live UC execution still needs a Databricks-accessible manifest/image directory
  and profile-authenticated compute.
- 2026-07-03: Ran live UC ingest against
  `serverless_stable_yau46e_catalog.cv_accuracy_levers`. Created schema
  `cv_accuracy_levers` and managed volume `cv_accuracy_levers` in the existing
  catalog, then ran
  `scripts/prepare_dataset.py` with `DATABRICKS_CONFIG_PROFILE=fevm`,
  `DATABRICKS_SERVERLESS_COMPUTE_ID=auto`,
  `--uc-image-upload-mode sdk`, `--write-uc`, and `--uc-table image_manifest`.
  Wrote image files under
  `/Volumes/serverless_stable_yau46e_catalog/cv_accuracy_levers/cv_accuracy_levers/artifacts/images`,
  wrote manifest JSON to
  `dbfs:/Volumes/serverless_stable_yau46e_catalog/cv_accuracy_levers/cv_accuracy_levers/artifacts/ingest/image_manifest`,
  and verified Delta table
  `serverless_stable_yau46e_catalog.cv_accuracy_levers.image_manifest` contains
  3 rows.
- 2026-07-03: Started Phase 5.1 threshold tuning lever. Scope is a sample-mode
  threshold sweep over existing baseline predictions, fixed-0.5 comparison, and
  MLflow logging; no new classifier, full-dataset training, or GPU work.
- 2026-07-05: Resumed after crash and completed Phase 5.1 threshold tuning
  verification. Verified local shared-code gate with
  `python -m compileall src tests scripts` and `pytest` with 40 tests passing.
  Verified local script and MLflow paths with
  `python scripts/tune_threshold.py --sample-mode true --runtime local_cpu` and
  `python scripts/tune_threshold.py --sample-mode true --runtime local_cpu --log-mlflow --tracking-uri file:/tmp/cv-accuracy-levers-threshold-mlruns`.
  Initial Databricks MLflow smoke failed without an experiment target, as
  expected for Databricks tracking; reran with
  `python scripts/tune_threshold.py --sample-mode true --runtime databricks_serverless_cpu --log-mlflow --tracking-uri databricks --experiment-name /Shared/cv-accuracy-levers`.
  The sandboxed Databricks attempt hit DNS resolution for the workspace host;
  the approved network rerun succeeded and logged an MLflow run. Final packaging
  passed with `DATABRICKS_AUTH_STORAGE=plaintext databricks bundle validate --profile fevm`,
  `DATABRICKS_AUTH_STORAGE=plaintext databricks bundle deploy --profile fevm`,
  and `DATABRICKS_AUTH_STORAGE=plaintext databricks bundle run threshold_tuning_sample_cpu --profile fevm`;
  the packaged serverless CPU job terminated `SUCCESS`.
- 2026-07-05: Started Phase 5.2 false-negative review grid. Scope is a
  sample-mode review artifact over existing baseline/threshold prediction
  scores; no new classifier, full-dataset training, embeddings, crop-first
  logic, or GPU work.
- 2026-07-05: Completed Phase 5.2 false-negative review grid. Added
  `src/levers/error_review.py`, `scripts/review_false_negatives.py`, a thin
  `notebooks/03_error_review_gut_check.py` wrapper, tests, README/plan updates,
  and the `false_negative_review_sample_cpu` bundle job. Verified local
  shared-code gate with `python -m compileall src tests scripts` and `pytest`
  with 48 tests passing. Verified script paths with
  `python scripts/review_false_negatives.py --sample-mode true --runtime local_cpu`
  and
  `python scripts/review_false_negatives.py --sample-mode true --runtime local_cpu --review-threshold 0.95`.
  The default selected-threshold review returned zero rows, as expected for the
  separated synthetic sample; the `0.95` review-threshold smoke returned one
  ranked false-negative row. Verified local MLflow artifact logging with
  `python scripts/review_false_negatives.py --sample-mode true --runtime local_cpu --review-threshold 0.95 --log-mlflow --tracking-uri file:/tmp/cv-accuracy-levers-error-review-mlruns`;
  the first attempt exposed MLflow's file-store opt-in requirement, so the
  script now sets `MLFLOW_ALLOW_FILE_STORE=true` for `file:` tracking URIs.
  Verified IDE-to-Databricks MLflow with
  `python scripts/review_false_negatives.py --sample-mode true --runtime databricks_serverless_cpu --review-threshold 0.95 --log-mlflow --tracking-uri databricks --experiment-name /Shared/cv-accuracy-levers`.
  The sandboxed attempt hit DNS resolution for the workspace host; the approved
  network rerun succeeded and logged an MLflow run. Local/IDE script context
  used runtime `local_cpu` or `databricks_serverless_cpu`, catalog
  `serverless_stable_yau46e_catalog`, schema `cv_accuracy_levers`, volume
  `cv_accuracy_levers`, and volume subpath `artifacts`. Final packaging passed
  with `DATABRICKS_AUTH_STORAGE=plaintext databricks bundle validate --profile fevm`;
  sandboxed deploy/run attempts hit DNS, approved network reruns of
  `DATABRICKS_AUTH_STORAGE=plaintext databricks bundle deploy --profile fevm`
  and
  `DATABRICKS_AUTH_STORAGE=plaintext databricks bundle run false_negative_review_sample_cpu --profile fevm`
  succeeded, and the packaged serverless CPU job terminated `SUCCESS` with
  runtime `databricks_serverless_cpu`, catalog `main`, schema
  `cv_accuracy_levers`, volume `cv_accuracy_levers`, and volume subpath
  `artifacts`.
- 2026-07-05: Started Phase 5.3 label-quality embeddings review. Scope is a
  CPU/sample-mode analysis artifact over deterministic baseline synthetic
  feature embeddings; no real image embedding model, CLIP/Torch dependency,
  crop-first logic, full-dataset extraction, or GPU work.
- 2026-07-05: Completed Phase 5.3 label-quality embeddings review. Added
  `src/levers/label_quality.py`,
  `scripts/review_label_quality_embeddings.py`, a thin
  `notebooks/04_label_quality_embeddings.py` wrapper, tests, README/plan
  updates, and the `label_quality_embeddings_sample_cpu` bundle job. Verified
  local shared-code gate with `python -m compileall src tests scripts` and
  `pytest` with 57 tests passing. Verified local script paths with
  `python scripts/review_label_quality_embeddings.py --sample-mode true --runtime local_cpu`
  and
  `python scripts/review_label_quality_embeddings.py --sample-mode true --runtime local_cpu --inject-synthetic-label-issue`.
  Verified local MLflow artifact logging with
  `python scripts/review_label_quality_embeddings.py --sample-mode true --runtime local_cpu --inject-synthetic-label-issue --log-mlflow --tracking-uri file:/tmp/cv-accuracy-levers-label-quality-mlruns`.
  Verified IDE-to-Databricks MLflow with
  `python scripts/review_label_quality_embeddings.py --sample-mode true --runtime databricks_serverless_cpu --inject-synthetic-label-issue --log-mlflow --tracking-uri databricks --experiment-name /Shared/cv-accuracy-levers`.
  The sandboxed Databricks attempt hit DNS resolution for the workspace host;
  the approved network rerun succeeded and logged an MLflow run with runtime
  `databricks_serverless_cpu`, catalog `serverless_stable_yau46e_catalog`,
  schema `cv_accuracy_levers`, volume `cv_accuracy_levers`, and volume subpath
  `artifacts`. Final packaging passed with
  `DATABRICKS_AUTH_STORAGE=plaintext databricks bundle validate --profile fevm`;
  sandboxed deploy/run attempts hit DNS, approved network reruns of
  `DATABRICKS_AUTH_STORAGE=plaintext databricks bundle deploy --profile fevm`
  and
  `DATABRICKS_AUTH_STORAGE=plaintext databricks bundle run label_quality_embeddings_sample_cpu --profile fevm`
  succeeded, and the packaged serverless CPU job terminated `SUCCESS` with
  runtime `databricks_serverless_cpu`, catalog `main`, schema
  `cv_accuracy_levers`, volume `cv_accuracy_levers`, and volume subpath
  `artifacts`. Synthetic injection produced 12 reviewed rows and is logged as
  smoke-test data only.
- 2026-07-05: Started Phase 5.4 crop-first A/B. Scope is a CPU/sample-only
  deterministic crop/region-emphasis comparison against the same whole-image
  baseline split; no real image crops, Torch, torchvision, OpenCV, GPU work, or
  new dataset dependency in this slice.
- 2026-07-05: Locally verified Phase 5.4 crop-first A/B shared code and script
  paths. `python -m compileall src tests scripts` passed, `pytest` passed with
  67 tests, `python scripts/run_crop_first_ab.py --sample-mode true --runtime local_cpu`
  passed, and
  `python scripts/run_crop_first_ab.py --sample-mode true --runtime local_cpu --log-mlflow --tracking-uri file:/tmp/cv-accuracy-levers-crop-first-mlruns`
  passed. The deterministic sample did not produce a recall gain:
  baseline recall was `1.0`, crop-first recall was `0.5`, and the review row
  flagged one new false negative for `synthetic/group_09/view_00.jpg`.
- 2026-07-05: Verified Phase 5.4 IDE-to-Databricks MLflow path with
  `python scripts/run_crop_first_ab.py --sample-mode true --runtime databricks_serverless_cpu --log-mlflow --tracking-uri databricks --experiment-name /Shared/cv-accuracy-levers`.
  The sandboxed attempt hit DNS resolution for the workspace host and was
  interrupted after showing `NameResolutionError`; the approved network rerun
  succeeded and logged run `79975b2ac7b249919a686e46ce6f2763` to experiment
  `1619757321015524` with runtime `databricks_serverless_cpu`, catalog
  `serverless_stable_yau46e_catalog`, schema `cv_accuracy_levers`, volume
  `cv_accuracy_levers`, and volume subpath `artifacts`.
- 2026-07-05: Completed Phase 5.4 crop-first A/B packaging gate. Added
  `crop_first_ab_sample_cpu` after local and IDE-to-Databricks verification
  passed. Verified `python -c "import yaml; yaml.safe_load(open('databricks.yml')); print('databricks_yml_ok')"`,
  `DATABRICKS_AUTH_STORAGE=plaintext databricks bundle validate --profile fevm`,
  `DATABRICKS_AUTH_STORAGE=plaintext databricks bundle deploy --profile fevm`,
  and
  `DATABRICKS_AUTH_STORAGE=plaintext databricks bundle run crop_first_ab_sample_cpu --profile fevm`.
  Sandboxed deploy/run attempts hit DNS for the workspace host; approved
  network reruns succeeded. The packaged serverless CPU job terminated
  `SUCCESS` and logged MLflow run `fa246d3e016f43fa94188e8f359f1291` to
  experiment `1619757321015524` with runtime `databricks_serverless_cpu`,
  catalog `main`, schema `cv_accuracy_levers`, volume `cv_accuracy_levers`,
  and volume subpath `artifacts`.
- 2026-07-05: Started Phase 6 AIR CLI-first GPU execution path after Phase 5.4
  completion was recorded. Added `src/levers/gpu_baseline.py`,
  `scripts/train_gpu_baseline.py`, `air/gpu_baseline_sample.yaml`, optional
  `gpu` dependencies, tests, and docs. The new path is manifest-backed and
  keeps `scripts/train_baseline.py` as the synthetic centroid CPU control.
  Locally verified with `python -m compileall src tests scripts`, `pytest`
  with 74 passed and 1 skipped, and
  `python -c "import yaml; yaml.safe_load(open('air/gpu_baseline_sample.yaml')); yaml.safe_load(open('air/baseline_sample.yaml')); print('air_yaml_ok')"`.
  The skipped test is the local CPU Torch/Torchvision smoke because those
  optional packages are not installed locally.
- 2026-07-05: Installed AIR CLI with
  `uv tool install --force databricks-air --python 3.12` and verified
  `air --version` reported Databricks AI Runtime CLI v0.1.0. Initial
  `air run --file air/gpu_baseline_sample.yaml --dry-run -p fevm` found that
  AIR v0.1.0 expects top-level `env_variables`, not nested
  `environment.env_variables`. After fixing the schema and changing
  `code_source.snapshot.root_path` to `..`, approved-network dry-run
  `air run --file air/gpu_baseline_sample.yaml --dry-run -p fevm --json`
  returned `DRY_RUN_OK` and packaged the repo root rather than only `air/`.
- 2026-07-05: Changed the default catalog for Phase 6 and repo defaults to
  `serverless_stable_yau46e_catalog` in `ProjectConfig`, `.env.example`,
  `databricks.yml`, AIR YAML, README, and persisted plan docs. Re-verified with
  `python -m compileall src tests scripts`, `pytest` with 74 passed and 1
  skipped, YAML parsing for `databricks.yml`, `air/gpu_baseline_sample.yaml`,
  and `air/baseline_sample.yaml`, and approved-network
  `air run --file air/gpu_baseline_sample.yaml --dry-run -p fevm --json`.
  The AIR dry-run returned `DRY_RUN_OK`; the generated payload showed
  `CV_CATALOG=serverless_stable_yau46e_catalog`, `GPU_1xA10`, repo-root
  snapshot packaging, and blank `CV_DATA_MANIFEST`/`CV_DATA_DIR`. Real
  `--watch` submission is pending Databricks-accessible manifest/image paths
  plus explicit approval because it starts GPU compute.
- 2026-07-05: Installed optional local GPU dependencies with
  `uv pip install -e '.[gpu]'`, which installed Torch `2.12.1` and Torchvision
  `0.27.1` into the project `.venv`. The shell `pytest` executable still uses
  the Miniconda interpreter, so its Torch smoke test remains skipped there, but
  the project `python` path verified direct local CPU Torch execution with
  `python scripts/train_gpu_baseline.py --manifest-path /tmp/cv-accuracy-levers-gpu-smoke/manifest.jsonl --data-dir /tmp/cv-accuracy-levers-gpu-smoke --sample-mode true --sample-size 6 --runtime local_cpu --device cpu --image-size 24 --batch-size 2 --epochs 1`.
  The smoke passed with runtime `local_cpu`, resolved device `cpu`, catalog
  `serverless_stable_yau46e_catalog`, recall `1.0`, precision `0.5`, F1
  `0.6666666666666666`, and false negatives `0`.
- 2026-07-06: Started Phase 6.1 real-data AIR GPU sample from commit
  `00d707f`. Scope is the first small real-image training run on Databricks
  Serverless GPU through `air run --watch`, using an existing permissively
  licensed manifest and image directory supplied outside git by
  `CV_DATA_MANIFEST` and `CV_DATA_DIR`. Keeping `sample_mode=true`,
  `GPU_1xA10`, `tiny_cnn`, one epoch, and no recall-improvement claim.
- 2026-07-06: Ran Phase 6.1 preflight. `python -m compileall src tests scripts`
  passed. `pytest` passed with 74 tests and 1 skipped optional Torch smoke
  under the shell interpreter. `air --version` reported Databricks AI Runtime
  CLI v0.1.0. `CV_DATA_MANIFEST` and `CV_DATA_DIR` were not set in the shell or
  `.env`, and a bounded search only found prior synthetic smoke manifests, not
  a real public dataset manifest. The planned real-data local CPU Torch smoke
  command
  `set -a; source .env; set +a; python scripts/train_gpu_baseline.py --manifest-path "$CV_DATA_MANIFEST" --data-dir "$CV_DATA_DIR" --sample-mode true --sample-size 32 --runtime local_cpu --device cpu --image-size 64 --batch-size 4 --epochs 1`
  failed with `ValueError: Set CV_DATA_MANIFEST or pass --manifest-path.` This
  is an input-data blocker, not a training-code failure.
- 2026-07-06: Validated AIR packaging for the intended UC manifest path with
  `DATABRICKS_AUTH_STORAGE=plaintext air run --file air/gpu_baseline_sample.yaml --dry-run -p fevm --json --override env_variables.CV_DATA_MANIFEST="/Volumes/serverless_stable_yau46e_catalog/cv_accuracy_levers/cv_accuracy_levers/artifacts/ingest/gpu_baseline_manifest.jsonl" env_variables.CV_DATA_DIR= env_variables.MLFLOW_EXPERIMENT_ID=""`.
  The sandboxed attempt retried workspace HTTPS and was interrupted with
  `INTERRUPTED`; the approved-network rerun returned `DRY_RUN_OK`. The payload
  used repo-root snapshot packaging, `GPU_1xA10`, runtime
  `ai_runtime_cli_gpu`, catalog `serverless_stable_yau46e_catalog`, schema
  `cv_accuracy_levers`, volume `cv_accuracy_levers`, volume subpath
  `artifacts`, `sample_mode=true`, and
  `/Volumes/serverless_stable_yau46e_catalog/cv_accuracy_levers/cv_accuracy_levers/artifacts/ingest/gpu_baseline_manifest.jsonl`.
  `DATABRICKS_AUTH_STORAGE=plaintext databricks fs ls dbfs:/Volumes/serverless_stable_yau46e_catalog/cv_accuracy_levers/cv_accuracy_levers/artifacts/ingest --profile fevm`
  initially hit sandbox DNS resolution; the approved-network rerun listed only
  the prior `image_manifest` entry, so `gpu_baseline_manifest.jsonl` is not
  staged. Skipped `scripts/prepare_dataset.py`, `databricks fs cp`, and
  `air run --watch` because the required local real-data source manifest and
  image directory were unavailable. No MLflow GPU run was created and no recall
  improvement is claimed.
- 2026-07-06: Created the Phase 6.1 real-image sample from the public
  `AI-Lab-Makerere/beans` Hugging Face dataset. The dataset API reported
  `license: mit`, which is accepted by the repo license guard. Downloaded
  `data/train-00000-of-00001.parquet`, `data/validation-00000-of-00001.parquet`,
  and `data/test-00000-of-00001.parquet` with `curl -L` into
  `data/beans/raw/`. Extracted 100 JPEGs into `data/beans/images` and wrote
  `data/beans/manifest.jsonl`, both under gitignored `data/`. The sample has
  60 train, 20 val, and 20 test rows; 68 defective rows and 32 normal rows.
  `healthy` maps to `normal`; `angular_leaf_spot` and `bean_rust` map to
  `defect`. The manifest is ordered round-robin by split and label so
  `--sample-size 32` includes both train/test classes and validation positives.
- 2026-07-06: Verified local CPU Torch training on the Beans real-image sample
  with
  `python scripts/train_gpu_baseline.py --manifest-path data/beans/manifest.jsonl --data-dir data/beans/images --sample-mode true --sample-size 32 --runtime local_cpu --device cpu --image-size 64 --batch-size 4 --epochs 1`.
  The run passed with runtime `local_cpu`, resolved device `cpu`, recall `1.0`,
  precision `0.5`, F1 `0.6666666666666666`, AUC PR `0.8261904761904761`, AUC
  ROC `0.8`, `comparison_status=needs_same_split_baseline_comparison`, and no
  recall-improvement claim.
- 2026-07-06: Staged the Beans sample to UC for AIR. Ran
  `DATABRICKS_AUTH_STORAGE=plaintext DATABRICKS_CONFIG_PROFILE=fevm DATABRICKS_SERVERLESS_COMPUTE_ID=auto python scripts/prepare_dataset.py --source manifest --manifest-path data/beans/manifest.jsonl --data-dir data/beans/images --output-path /tmp/cv-accuracy-levers-gpu-real-manifest.jsonl --sample-mode true --sample-size 100 --runtime databricks_serverless_cpu --catalog serverless_stable_yau46e_catalog --schema cv_accuracy_levers --volume cv_accuracy_levers --volume-subpath artifacts --copy-images-to-uc-volume --uc-image-upload-mode sdk --databricks-profile fevm --write-uc --uc-table image_manifest`.
  The sandboxed run hit DNS resolution for the workspace host. The
  approved-network rerun uploaded images and wrote the normalized local
  manifest, then failed only at the UC Delta table write because the project
  `.venv` does not have `pyspark` or `databricks-connect` installed. The
  normalized manifest contains 100 rows with image paths under
  `/Volumes/serverless_stable_yau46e_catalog/cv_accuracy_levers/cv_accuracy_levers/artifacts/images`.
  Verified the UC image upload with
  `DATABRICKS_AUTH_STORAGE=plaintext databricks fs ls dbfs:/Volumes/serverless_stable_yau46e_catalog/cv_accuracy_levers/cv_accuracy_levers/artifacts/images --profile fevm`
  and a representative nested image listing.
- 2026-07-06: Copied the normalized AIR manifest to UC with
  `DATABRICKS_AUTH_STORAGE=plaintext databricks fs cp /tmp/cv-accuracy-levers-gpu-real-manifest.jsonl dbfs:/Volumes/serverless_stable_yau46e_catalog/cv_accuracy_levers/cv_accuracy_levers/artifacts/ingest/gpu_baseline_manifest.jsonl --overwrite --profile fevm`.
  The sandboxed copy hit DNS resolution; the approved-network rerun succeeded.
  Verified
  `dbfs:/Volumes/serverless_stable_yau46e_catalog/cv_accuracy_levers/cv_accuracy_levers/artifacts/ingest/gpu_baseline_manifest.jsonl`
  exists alongside the prior `image_manifest` entry. Re-ran AIR dry-run with
  `DATABRICKS_AUTH_STORAGE=plaintext air run --file air/gpu_baseline_sample.yaml --dry-run -p fevm --json --override env_variables.CV_DATA_MANIFEST="/Volumes/serverless_stable_yau46e_catalog/cv_accuracy_levers/cv_accuracy_levers/artifacts/ingest/gpu_baseline_manifest.jsonl" env_variables.CV_DATA_DIR= env_variables.MLFLOW_EXPERIMENT_ID=""`
  and received `DRY_RUN_OK` with `GPU_1xA10` and repo-root snapshot packaging.
  AIR should use
  `CV_DATA_MANIFEST=/Volumes/serverless_stable_yau46e_catalog/cv_accuracy_levers/cv_accuracy_levers/artifacts/ingest/gpu_baseline_manifest.jsonl`
  and blank `CV_DATA_DIR`. GPU `--watch` submission has not been started yet.
- 2026-07-06: Installed the existing repo Databricks extra into the project
  `.venv` with `uv pip install -e '.[databricks]'`, which installed
  `databricks-connect==16.1.7` plus Spark Connect dependencies. Verified
  imports for `databricks.connect`, `pyspark`, and `databricks.sdk`. Also added
  the matching README note for IDE UC ingest setup.
- 2026-07-06: Re-ran the Beans sample UC ingest with `--write-uc` after
  installing Databricks Connect. The sandboxed attempt reproduced the workspace
  DNS blocker; the approved-network rerun of
  `DATABRICKS_AUTH_STORAGE=plaintext DATABRICKS_CONFIG_PROFILE=fevm DATABRICKS_SERVERLESS_COMPUTE_ID=auto python scripts/prepare_dataset.py --source manifest --manifest-path data/beans/manifest.jsonl --data-dir data/beans/images --output-path /tmp/cv-accuracy-levers-gpu-real-manifest.jsonl --sample-mode true --sample-size 100 --runtime databricks_serverless_cpu --catalog serverless_stable_yau46e_catalog --schema cv_accuracy_levers --volume cv_accuracy_levers --volume-subpath artifacts --copy-images-to-uc-volume --uc-image-upload-mode sdk --databricks-profile fevm --write-uc --uc-table image_manifest`
  passed. It wrote 100 records, 100 groups, 68 positives, 32 negatives, 60
  train, 20 val, and 20 test rows. UC outputs were
  `serverless_stable_yau46e_catalog.cv_accuracy_levers.image_manifest` and
  `dbfs:/Volumes/serverless_stable_yau46e_catalog/cv_accuracy_levers/cv_accuracy_levers/artifacts/ingest/image_manifest`.
  A Databricks Connect SQL count verified `image_manifest_count=100`,
  `image_manifest_positives=68`, and `image_manifest_negatives=32`.
- 2026-07-06: Re-verified local gates after installing Databricks Connect.
  `python -m compileall src tests scripts` passed. Shell `pytest` passed with
  74 tests and 1 optional Torch smoke skipped under the Miniconda interpreter.
  The project `.venv` lacked pytest, so installed the existing dev extra with
  `uv pip install -e '.[dev,databricks]'`; then `python -m pytest` passed with
  75 tests under Python 3.11.13, including the local Torch smoke.
