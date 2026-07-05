# AI Runtime CLI Readiness

AI Runtime CLI is the Phase 6 GPU path for this repo. Use it after the
sample-mode baseline and Phase 5 levers have comparable MLflow runs on the same
split. Notebook GPU work should stay as a thin optional wrapper after the
script/library path is stable.

`baseline_sample.yaml` is a historical CPU-style workload contract. The active
Phase 6 GPU contract is `gpu_baseline_sample.yaml`, which runs
`scripts/train_gpu_baseline.py` over a public image manifest with Torch,
Torchvision, Pillow, MLflow, and `GPU_1xA10`.

As of 2026-07-05, AIR CLI v0.1.0 is installed locally and `air --version`
passes. Re-run `air --version` and the dry-run after CLI upgrades.

Expected sequence:

```bash
python scripts/train_gpu_baseline.py \
  --manifest-path "$CV_DATA_MANIFEST" \
  --data-dir "$CV_DATA_DIR" \
  --sample-mode true \
  --sample-size 32 \
  --runtime local_cpu \
  --device cpu

air --help
air run --file air/gpu_baseline_sample.yaml --dry-run -p <profile> \
  --override env_variables.CV_DATA_MANIFEST="$CV_DATA_MANIFEST" \
             env_variables.CV_DATA_DIR="$CV_DATA_DIR" \
             env_variables.MLFLOW_EXPERIMENT_ID="$MLFLOW_EXPERIMENT_ID"
air run --file air/gpu_baseline_sample.yaml --watch -p <profile> \
  --override env_variables.CV_DATA_MANIFEST="$CV_DATA_MANIFEST" \
             env_variables.CV_DATA_DIR="$CV_DATA_DIR" \
             env_variables.MLFLOW_EXPERIMENT_ID="$MLFLOW_EXPERIMENT_ID"
```

The AIR run must log `CV_RUNTIME=ai_runtime_cli`, `SAMPLE_MODE=true`,
catalog/schema/volume settings, manifest path, sample size, lever name, GPU
config, backbone, image size, epochs, batch size, selected threshold, and the
same defective-class metrics used by local and serverless CPU runs. It must
also log `predictions.json`, `threshold_sweep.json`, `training_summary.json`,
and `leaderboard_row.json`.

Use `GPU_1xA10` as the default small CV demo accelerator. Keep multi-GPU and
distributed work deferred.

AIR CLI v0.1.0 does not interpolate local shell variables inside checked-in
`env_variables`; pass real run values with `--override env_variables.<name>=...`.
