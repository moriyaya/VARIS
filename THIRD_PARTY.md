# Dependencies and attribution

- **IKFlow**: https://github.com/jstmn/ikflow. Panda response generation uses its externally pretrained model. The upstream source and model are downloaded separately, with checksums; their original notices and terms are retained.
- **JRL**: https://github.com/jstmn/Jrl. Robot kinematics, model assets and collision checking use the pinned upstream revision in `configs/robotics_assets.json`.
- **QUBIQ**: https://qubiq.grand-challenge.org/. Medical images and expert annotations are not redistributed.
- PyTorch, NumPy, SciPy, NiBabel, Matplotlib and the optional robotics dependencies retain their respective licenses.

`varis/optimization.py` contains the implicit-hypergradient, conjugate-gradient and weighted-Chebyshev simplex-QP helpers from the project's shared WC-MHGD implementation. VARIS uses these primitives for the upper updates in the toy/Panda studies. These shared primitives implement the upper-level updates.
