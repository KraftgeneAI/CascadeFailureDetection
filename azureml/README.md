# MLOps — Stage 1: Track & Version

This folder turns the existing `train_model.py` into a **tracked, reproducible**
training run. Nothing about how you train changes — the instrumentation is a
no-op until you install MLflow and point it somewhere.

## What Stage 1 gives you

- **Experiment tracking** — every run records its params (all CLI args, model
  size, class balance, calibrated loss weights), per-epoch metrics (cascade/node
  F1, precision, recall, timing MAE, risk MSE, LR) and artifacts (training
  curves, history JSON, best checkpoints).
- **Data versioning** — each run is tagged with a `data_version` fingerprint of
  the exact `data/train` + `data/val` files it trained on, plus the `git_sha` of
  the code. This is what would have caught the "feature widths changed, old
  checkpoint silently invalid" problem.
- **Checkpoints out of git** — `checkpoints/*.pth` are now git-ignored; model
  binaries belong in the model registry, not the repo.

## Run it locally (zero cloud)

```bash
pip install -r requirements.txt -r azureml/requirements-mlops.txt

# Train — logs to a local ./mlruns store
python train_model.py --data_dir data --epochs 60

# Browse the runs
mlflow ui        # http://127.0.0.1:5000
```

Disable tracking anytime with `MLFLOW_DISABLE=1`.

## Point the same run at Azure ML

No code change — just set the tracking URI to your workspace:

```bash
az login
az ml workspace show -n <workspace> -g <resource-group> \
    --query mlflow_tracking_uri -o tsv         # copy this
export MLFLOW_TRACKING_URI="azureml://..."     # (the value above)

python train_model.py --data_dir data          # runs now appear in Azure ML Studio
```

## Run training *on* Azure ML compute

```bash
# one-time: register the environment and a versioned data asset
az ml environment create -f azureml/environment.yml
az ml data create -f azureml/data.yml --set version=$(date +%Y%m%d)

# submit the GPU training job (MLflow tracking is wired automatically)
az ml job create -f azureml/train-job.yml --web
```

Set `DATA_VERSION` (env var) to override the auto fingerprint with a human label
(e.g. the Azure ML data asset version) so runs and data assets line up.

## Files

| File | Purpose |
|------|---------|
| `conda.yml` | Pinned training environment (mirrors `requirements.txt` + MLOps extras) |
| `environment.yml` | Azure ML Environment asset referencing `conda.yml` |
| `data.yml` | Azure ML Data asset for the scenario set (`cascade-scenarios`) |
| `train-job.yml` | Azure ML command job that runs `train_model.py` on a GPU cluster |
| `requirements-mlops.txt` | `mlflow`, `azure-ai-ml`, `azureml-mlflow` |

## Next (Stage 2, not yet built)

Pipeline of components (generate → train → evaluate → register-if-better) with a
**schema/contract check** as the first step and an **eval gate** that only
registers a model when it beats the current champion on cascade F1.
