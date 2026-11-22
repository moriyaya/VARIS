# VARIS

**Validity Before Preference: Resolving Lower-Level Response Ambiguity in Bilevel Multi-Objective Optimization**

Jiaxin Gao · Zhenyu Liang · Ran Cheng

## Overview

Equally optimal lower-level responses can induce different upper-level outcomes, a phenomenon we call **lower-level response ambiguity**. VARIS (**Validity-Aware Response-Set Inference and Selection**) addresses this ambiguity through the principle of **validity before preference**:

1. Infer a preference-independent candidate response set in one forward pass.
2. Admit candidates using a task-derived lower-level validity criterion.
3. Apply upper preference only to admitted responses.

The same response set serves multiple preference queries, reducing repeated lower-level solves. Optional safeguarded local repair can recover near-valid candidates when needed.

## Experiments

A numerical toy study examines the response-selection mechanism. Three complementary applications assess solution quality, robustness, and inference efficiency.

| Task | Benchmark |
|---|---|
| Numerical toy | Frozen-scale and decision-coupled triple wells |
| Robotic configuration optimization | Panda robot |
| PDE inverse problems | Nonlinear Allen–Cahn system |
| Medical image segmentation | QUBIQ prostate segmentation |

## Getting started

Use Python 3.10–3.12. Run the following commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Evaluate the included checkpoints on a small query set.
python -m varis.toy.run evaluate --quick --output outputs/toy
python -m varis.pde.run evaluate --quick --output outputs/pde
```

The repository includes VARIS implementations, experiment configurations, and nine trained toy/PDE checkpoints.

- [Training and evaluation](docs/RUNNING.md)
- [Experiment settings and metrics](docs/PROTOCOLS.md)
- [Implementation checks](VALIDATION.md)

## License and acknowledgments

Released under the [MIT License](LICENSE). See [third-party acknowledgments](THIRD_PARTY.md) for IKFlow, JRL, QUBIQ, and optimization dependencies. Citation metadata is available in [CITATION.cff](CITATION.cff).
