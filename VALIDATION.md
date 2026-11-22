# Release validation

Validation date: 22 November 2026. All checks below used CPU inference. They validate this extracted release; they are not new GPU latency measurements or reruns of all training experiments.

| Check | Scope | Outcome |
|---|---|---|
| Numerical toy evaluation | Both variants, three supplied checkpoints each, 42 queries per checkpoint (252 total) | Completed; all returned responses valid and preferred-valid in these runs |
| PDE evaluation | Three checkpoints × three splits × 128 tasks × four preferences (4,608 queries) | Completed; all responses admitted; preference/branch outcomes retained separately |
| Original/release equivalence | PDE low-extrapolation seed0, 512 responses including fallback; one complete toy query | Response tensors exactly equal; PDE stage labels also equal |
| Training entry points | Two toy optimization steps; shortened PDE training covering distillation, lower loss and fine-tuning | Completed with finite losses |
| Segmentation training/evaluation | QUBIQ task02, fold0/seed0, one epoch on real data, validation selection then test evaluation | Completed; this is a functional check, not the 300-epoch paper result |
| Robotics | One existing target, three preferences, two upper updates, original frozen IKFlow assets | Completed with valid returned responses |
| Mathematical checks | Exact toy optima, set-permutation invariance, reflected PDE validity, preference/repair trigger, segmentation gradient | Four automated tests passed |
| Plotting | Toy response-set example | PDF generated |

Toy, PDE and segmentation were checked with Python 3.10, PyTorch 2.5.1, NumPy 1.26.3, SciPy 1.14.1 and NiBabel 5.3.2. Robotics used the existing Linux environment with PyTorch 2.5.0, NumPy 1.24.4 and SciPy 1.14.1. The installation file pins PyTorch 2.5.1 and NumPy 1.26.4 for a fresh environment; a completely fresh installation of every optional robotics dependency was not rerun.

The full segmentation 5-fold × 3-seed training and the full robotics target grid were not rerun during release packaging. Their configurations and execution paths are included. The provided toy/PDE checkpoints were extracted from the archived models without changing their weights.
