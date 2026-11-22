# Experiment settings

| Task | Evaluation settings | Configuration |
|---|---|---|
| Numerical toy | Two triple-well variants; six contexts; seven preferences; three seeds; validity radius 0.18 | `configs/toy.json` |
| Robotic configuration optimization | 32 Panda targets; three preferences; three seeds; 80 upper updates at step size 0.1 | `configs/panda.json` |
| PDE inverse problems | Three diffusion ranges; 128 tasks per range and seed; four preferences; three seeds; validity tolerance 1e-4 | `configs/pde.json` |
| Medical image segmentation | 48 QUBIQ cases; five folds; three seeds; six responses; 128 × 128 images | `configs/qubiq.json`, `configs/qubiq_splits.json` |

Training and evaluation commands are in [RUNNING.md](RUNNING.md).

## Metrics

**Numerical toy:** `preferred_valid` is PVR. `lower_gap` is squared distance to the exact response set; `projected_regret` evaluates the nearest exact response at the returned upper decision. The decision-coupled study also reports `upper_decision_error`.

**Robotics:** each query returns the best preference-weighted member of its valid archive. Admission requires position error ≤1 mm, orientation error ≤1°, joint-limit compliance and no self-collision.

**PDE:** `joint` is preferred-valid success, requiring observation MSE ≤1e-4 and preference-consistent orientation. `valid` reports observation consistency. Preferences use weights `(1-r, r)`; robotics uses `(r, 1-r)`.

**Segmentation:** Q-Dice averages thresholds 0.1–0.9; Precision Dice averages each prediction's best expert overlap. Conservative, central and sensitive Dice compare independently selected predicted and expert responses. Annotation-coverage Dice selects the validation checkpoint. Generation latency reports the mean and sample standard deviation of five block medians, each formed from the 48 case-level medians.

For segmentation, pool all out-of-fold cases within each seed. For the other tasks, average queries within each seed and evaluation split. Report the mean and sample standard deviation across seeds.
