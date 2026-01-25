# Repository Guidelines

## Project Structure & Module Organization
- Core training and preprocessing live in top-level scripts: `train.py`, `BCIC2020Track3_preprocess.py`, `plot_accuracies.py`.
- Model definitions are in `FAST.py` (transformer) and `FAST_mamba.py` (Mamba2); shared helpers are in `utils.py`.
- Raw data is expected under `BCIC2020Track3/` with `Training set/` and `Validation set/` subfolders.
- Generated artifacts are written to `Processed/` (H5 cache) and `Results/` (per-run metrics); both are gitignored.

## Build, Test, and Development Commands
- Install deps (preferred): `uv sync` (uses `uv.lock`). Optional build tuning for mamba-ssm: `USE_NINJA=1 MAX_JOBS=8 uv sync`.
- Preprocess data: `python BCIC2020Track3_preprocess.py` (writes `Processed/BCIC2020Track3.h5`).
- Train a run: `python train.py --model transformer --accelerator gpu --gpu 0 --folds "0-15"`.
- Full sweep: `bash sweep.sh` (runs transformer + mamba2 variants).
- Plot results: `python plot_accuracies.py` (writes `Results/model_accuracies_comparison.png`).
- Multi-GPU: `bash BCIC2020Track3_run.sh` (splits folds across two GPUs).

## Coding Style & Naming Conventions
- Python, 4-space indentation, PEP8-ish formatting.
- `snake_case` for functions/variables, `CapWords` for classes, `UPPER_SNAKE_CASE` for module constants.
- No formatter/linter is enforced; keep changes tight and consistent with nearby code.

## Testing Guidelines
- No automated test suite is present.
- For quick sanity checks, run a single fold: `python train.py --folds "0"` (after preprocessing) and confirm `Results/.../accuracies.csv` is produced.

## Commit & Pull Request Guidelines
- Commit messages in history are short, lowercase phrases; follow that style unless a change needs extra detail.
- PRs should include: a brief summary, commands run, and any relevant metrics or plots.
- Do not commit datasets or outputs: `BCIC2020Track3/`, `Processed/`, `Results/` are intentionally ignored.
