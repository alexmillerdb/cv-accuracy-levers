# AI Runtime CLI Readiness

AI Runtime CLI is a deferred GPU path for this repo. Use it only after the
sample-mode baseline and Phase 5 levers have comparable MLflow runs on the same
split, and after a notebook-first GPU smoke has passed.

The checked-in YAML is an example workload contract, not a current verification
gate. Databricks currently documents single-node AI Runtime as Public Preview,
multi-GPU distributed APIs as Beta, and the `air` CLI as Beta. Before submitting
this example, validate the file against the installed `air` CLI version and
workspace policy.

Expected sequence:

```bash
python scripts/train_baseline.py --sample-mode true --runtime local_cpu
databricks bundle run baseline_sample_cpu --profile <profile>
# Run the Phase 6 notebook-first GPU smoke before AIR CLI work.
air --help
air run --file air/baseline_sample.yaml --dry-run
air run --file air/baseline_sample.yaml
```

The AIR run must log `CV_RUNTIME=ai_runtime_cli`, `SAMPLE_MODE=true`,
catalog/schema/volume settings, sample size, lever name, and the same baseline
metrics used by local and serverless CPU runs.

Use `1xA10` as the default small CV demo accelerator, `1xH100` only when memory
bound, and `8xH100` only for explicit multi-GPU Beta work.
