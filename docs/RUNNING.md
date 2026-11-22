# Training and evaluation

Run commands from the repository root after installing `requirements.txt`. For GPU execution, use a PyTorch build matching your CUDA environment. Each run writes to its specified output directory.

## Numerical toy

Evaluate the supplied generators on six contexts and seven preferences:

```bash
for seed in 0 1 2; do
  python -m varis.toy.run evaluate --seed "$seed" --output "outputs/toy/frozen_$seed"
  python -m varis.toy.run decision --variant coupled --seed "$seed" --output "outputs/toy/coupled_$seed"
done
```

Train a generator and plot its responses:

```bash
python -m varis.toy.run train --variant frozen --seed 0 --output outputs/toy/train_frozen
python -m varis.toy.run train --variant coupled --seed 0 --output outputs/toy/train_coupled
python -m varis.toy.run decision --variant coupled --checkpoint outputs/toy/train_coupled/generator.pt --output outputs/toy/retrained
python -m scripts.plot_toy --output outputs/toy_responses.pdf
```

Use the bundled checkpoints to evaluate the archived models. Standalone retraining uses the same architecture, objective and schedule with a different initialization sequence from the original joint experiment. Results are saved as `results.csv` and `summary.json`.

## Robotic configuration optimization

Download the pretrained IKFlow model and pinned IKFlow/JRL sources:

```bash
python scripts/fetch_robotics.py
python -m pip install -r requirements-robotics.txt
python -m pip install --no-deps -e assets/robotics/vendor/jrl
python -m pip install --no-deps -e assets/robotics/vendor/ikflow
CUDA_VISIBLE_DEVICES=0 python -m varis.robotics.run --seed 0 --output outputs/panda/seed0
```

Repeat for seeds 1 and 2. Expose one GPU per process; set `CUDA_VISIBLE_DEVICES=""` for CPU. The frozen generator produces candidate configurations, followed by physical admission, preference selection and local repair. Settings and asset checksums are in `configs/panda.json` and `configs/robotics_assets.json`. Per-query JSON files contain the selected response, valid archive and timing.

## PDE inverse problems

```bash
for seed in 0 1 2; do
  python -m varis.pde.run evaluate --seed "$seed" --device cpu --output "outputs/pde/seed$seed"
done
python -m varis.pde.run train --seed 0 --device cuda:0 --output outputs/pde/training
python -m varis.pde.run evaluate --checkpoint outputs/pde/training/generator.pt --device cuda:0 --output outputs/pde/retrained
```

The included Allen–Cahn solver generates the training data and observations. `configs/pde.json` specifies training and evaluation settings. Each evaluation covers 128 tasks per split and four preferences, saving per-query results and split-level summaries.

## Medical image segmentation

Obtain the prostate training data from the [QUBIQ challenge](https://qubiq.grand-challenge.org/). Point `--data` to the directory containing `case01` through `case48`, each with `image.nii.gz` and `task02_seg*.nii.gz`.

```bash
python -m varis.qubiq.run train --data /path/to/prostate --fold 0 --seed 0 --device cuda:0 --output outputs/qubiq/fold0_seed0
python -m varis.qubiq.run evaluate --data /path/to/prostate --fold 0 --seed 0 --device cuda:0 --checkpoint outputs/qubiq/fold0_seed0/best.pt --output outputs/qubiq/eval_fold0_seed0
```

Repeat folds 0–4 and seeds 0–2. This application uses six-response generation and prediction-based preference selection. The supplied splits and training settings are in `configs/qubiq_splits.json` and `configs/qubiq.json`. Checkpoints maximize validation annotation-coverage Dice; `result.json` contains case-level evaluation metrics.

## Implementation checks

```bash
python -m unittest discover -s tests -v
```

For shorter runs, use `--quick` (toy/PDE), `--targets 1 --steps 2` (robotics), or `--epochs 1` (segmentation). Completed checks and their budgets are recorded in [VALIDATION.md](../VALIDATION.md).

The package contains VARIS code and checkpoints; comparison methods and ablation sweeps are excluded. External datasets and large pretrained assets are obtained using the instructions above.
