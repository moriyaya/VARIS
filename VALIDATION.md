# Implementation checks

CPU checks completed on 22 November 2026:

| Check | Evaluated scope | Result |
|---|---|---|
| Numerical toy | Two variants × three seeds × 42 queries | Completed |
| PDE inverse problems | Three seeds × three splits × 128 tasks × four preferences | Completed |
| Consistency with original implementation | 512 PDE responses and stage labels; one complete toy query | Identical outputs |
| Training entry points | Two toy steps; shortened PDE distillation, lower-loss training and fine-tuning | Passed |
| Medical image segmentation | QUBIQ fold0/seed0, one training epoch, validation selection and test evaluation | Passed |
| Robotic configuration optimization | One target, three preferences, two upper updates | Passed |
| Mathematical and gradient checks | Four automated tests | Passed |
| Plotting | Toy response-set example | PDF generated |

Toy, PDE and segmentation: Python 3.10, PyTorch 2.5.1, NumPy 1.26.3, SciPy 1.14.1, NiBabel 5.3.2. Robotics: PyTorch 2.5.0, NumPy 1.24.4, SciPy 1.14.1, with the existing IKFlow/JRL environment.
