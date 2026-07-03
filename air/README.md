# AI Runtime CLI Readiness

AI Runtime CLI is a deferred GPU path for this repo. Use it only after the
sample-mode baseline and at least one lever have comparable MLflow runs on the
same split.

The checked-in YAML is an example workload contract, not a current verification
gate. Before submitting it, validate the file against the installed `air` CLI
version and workspace policy.

Expected sequence:

```bash
python scripts/train_baseline.py --sample-mode true --runtime local_cpu
databricks bundle run baseline_sample_cpu --profile <profile>
air --help
air run --file air/baseline_sample.yaml --dry-run
air run --file air/baseline_sample.yaml
```

The AIR run must log `CV_RUNTIME=ai_runtime_cli`, `SAMPLE_MODE=true`,
catalog/schema/volume settings, sample size, lever name, and the same baseline
metrics used by local and serverless CPU runs.
