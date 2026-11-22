# Dependencies

- **IKFlow** supplies the pretrained Panda response generator; **JRL** provides kinematics and collision checking. Pinned sources and model assets are specified in `configs/robotics_assets.json`. Their upstream licenses and notices are retained by the downloader.
- **QUBIQ** provides the prostate images and expert annotations, subject to the dataset terms.
- PyTorch, NumPy, SciPy, NiBabel, Matplotlib and the optional robotics libraries retain their respective licenses.

The shared upper-optimization utilities implement implicit hypergradients, conjugate gradients and a weighted-Chebyshev preference QP.
