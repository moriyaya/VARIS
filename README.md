# VARIS

**Validity Before Preference: Resolving Lower-Level Response Ambiguity in Bilevel Multi-Objective Optimization**

VARIS (**Validity-Aware Response-Set Inference and Selection**) predicts a preference-independent set of lower-level responses, checks their lower-level admissibility, and ranks admitted responses by upper preference. Safeguarded local repair supplements direct selection where needed.

This compact research release contains VARIS implementations for the numerical toy and the three applications in the manuscript. It includes training/evaluation configurations, small toy/PDE checkpoints, and one plotting example. Comparison methods and ablation sweeps are not included; the package reproduces the VARIS paths rather than complete comparative tables.

## Included experiments

| Experiment | Implementation | Assets |
|---|---|---|
| Numerical toy | Three-response MLP; frozen-scale and decision-coupled triple wells | Six trained generator checkpoints included |
| Robotic configuration optimization | Frozen IKFlow, physical admission, repair, upper optimization | External IKFlow/JRL sources and checkpoint; verified downloader |
| PDE inverse problems | Conditional response generator, Allen–Cahn forward model, preference selection and safeguarded repair | Three trained generator checkpoints included |
| Medical image segmentation | Six-head compact U-Net, Hungarian set matching, prediction-based preference selection | QUBIQ prostate task02 data obtained separately; fixed five-fold splits included |

The segmentation study evaluates expert-set coverage and preference-aware prediction. It does not use an online lower-validity certificate or local repair. The other tasks implement their task-specific numerical/physical checks.

## Installation

Use Python 3.10–3.12 in a dedicated environment. Run commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For CUDA, install the matching PyTorch build for your system before installing the other dependencies. CPU execution is supported for toy, PDE, and segmentation. Formal segmentation training and robotics are best run on a Linux machine with a GPU. Record hardware when measuring latency; CPU smoke-test times do not reproduce GPU timing claims.

## Quick verification

```bash
python -m varis.toy.run evaluate --quick --output outputs/toy_smoke
python -m varis.toy.run decision --variant coupled --quick --output outputs/coupled_smoke
python -m varis.pde.run evaluate --quick --output outputs/pde_smoke
python -m unittest discover -s tests -v
```

These short commands verify execution, not the complete experimental protocol. See [VALIDATION.md](VALIDATION.md) for the checks actually performed on this release.

## 1. Numerical toy

Evaluate the included checkpoints on the six contexts and seven preferences:

```bash
for seed in 0 1 2; do
  python -m varis.toy.run evaluate --seed "$seed" --output "outputs/toy/frozen_$seed"
  python -m varis.toy.run decision --variant coupled --seed "$seed" --output "outputs/toy/coupled_$seed"
done
```

Train a new VARIS generator with the original set-matching objective:

```bash
python -m varis.toy.run train --variant frozen --seed 0 --output outputs/toy/train_frozen
python -m varis.toy.run train --variant coupled --seed 0 --output outputs/toy/train_coupled
python -m varis.toy.run decision --variant coupled --checkpoint outputs/toy/train_coupled/generator.pt --output outputs/toy/retrained
python -m scripts.plot_toy --output outputs/toy_responses.pdf
```

The released checkpoints are the VARIS weights from the archived experiments. The standalone training entry initializes only VARIS, whereas the original joint experiment initialized comparison models as well. Consequently, retraining preserves the architecture, objective and schedule but does not reproduce the original random-number consumption byte for byte. Use the included checkpoints to evaluate the archived models.

Outputs contain per-query `results.csv` and a `summary.json`. `preferred_valid` means the returned response satisfies the prescribed radius and its nearest exact branch is preference-optimal (ties accepted). `projected_regret` evaluates the nearest exact response, whereas `raw_regret` evaluates the actual returned response. The coupled study also reports upper-decision error.

## 2. Robotic configuration optimization

The experiment uses a **frozen, externally pretrained** IKFlow generator. It does not train IKFlow. Download its original assets and install the dependencies:

```bash
python scripts/fetch_robotics.py
python -m pip install -r requirements-robotics.txt
python -m pip install --no-deps -e assets/robotics/vendor/jrl
python -m pip install --no-deps -e assets/robotics/vendor/ikflow
CUDA_VISIBLE_DEVICES=0 python -m varis.robotics.run --seed 0 --output outputs/panda/seed0
```

Repeat with seeds 1 and 2 for the full protocol. Expose one GPU per process (`CUDA_VISIBLE_DEVICES=0`); use `CUDA_VISIBLE_DEVICES=""` for CPU. `configs/panda.json` fixes 32 evaluation targets, three preferences and 80 upper updates. Target generation samples the original 128-target bank before selecting its first 32 cases; this preserves the sampling order. Pose witnesses are excluded from the response oracle. The checker uses position error, orientation error, joint bounds and mesh self-collision. Per-query JSON files contain the valid archive, selected response, final state, timing and repair/fallback counts.

`--targets 1 --steps 2` is a functional check only. The external asset checksums and pinned IKFlow/JRL commits are in `configs/robotics_assets.json`. The downloader verifies each SHA-256 before extraction. Dependency licenses remain with their authors.

## 3. PDE inverse problems

```bash
for seed in 0 1 2; do
  python -m varis.pde.run evaluate --seed "$seed" --device cpu --output "outputs/pde/seed$seed"
done
python -m varis.pde.run train --seed 0 --device cuda:0 --output outputs/pde/training
python -m varis.pde.run evaluate --checkpoint outputs/pde/training/generator.pt --device cuda:0 --output outputs/pde/retrained
```

Each full evaluation uses 128 tasks for each of interpolation, low extrapolation, and high extrapolation, with four preferences. The training entry implements the initial 1,000 distillation + 1,000 lower-loss steps, followed by the archived 5,000-step fine-tuning stage. Training data and observations are simulated by the included forward solver; there is no external dataset.

The admission threshold is observation MSE ≤ 1e-4. `valid` reports this check; `branch_correct` reports agreement with the preferred reference branch, and `joint` requires both. Reference teachers are used only to calculate evaluation metrics. A five-start lower solver is retained as VARIS's fallback when repair yields no admitted response; it is not run as a separate comparison method.

## 4. Medical image segmentation

Obtain the QUBIQ prostate training data under its dataset terms from the [QUBIQ challenge](https://qubiq.grand-challenge.org/). This release contains no patient images or expert masks. Point `--data` to the directory containing `case01` through `case48`; each case contains `image.nii.gz` and `task02_seg*.nii.gz`.

```bash
python -m varis.qubiq.run train --data /path/to/prostate --fold 0 --seed 0 --device cuda:0 --output outputs/qubiq/fold0_seed0
python -m varis.qubiq.run evaluate --data /path/to/prostate --fold 0 --seed 0 --device cuda:0 --checkpoint outputs/qubiq/fold0_seed0/best.pt --output outputs/qubiq/eval_fold0_seed0
```

Repeat folds 0–4 and seeds 0–2. The supplied splits match the archived five-fold protocol. `configs/qubiq.json` specifies 300 epochs, AdamW, augmentation, batch size and validation cadence. Checkpoints maximize validation annotation-coverage Dice; test masks do not select checkpoints or predicted responses. Evaluation reports annotation coverage, prediction precision, generalized energy distance, diversity, Q-Dice and preference Dice. Conservative, central and sensitive responses are selected from predictions by area and medoid rules; reference preferences are computed separately for scoring. See [docs/PROTOCOLS.md](docs/PROTOCOLS.md) for metric aggregation.

## Reproducibility and scope

- Run each experiment in its own output directory. Defaults are protocol settings; `--quick`, `--steps`, `--epochs` and `--targets` overrides create reduced runs.
- The paper's comparative tables require the separately maintained comparison implementations. They are intentionally outside this VARIS-only release.
- Datasets and large external model assets are not bundled. Segmentation checkpoints are produced by the provided training command.
- [CHECKSUMS.sha256](CHECKSUMS.sha256) identifies the distributed files. [THIRD_PARTY.md](THIRD_PARTY.md) records external dependencies and shared optimization primitives.

## License

The authors' code is released under the [MIT License](LICENSE). External code, pretrained models, and datasets remain subject to their respective terms.
