# Method and implementation references

These references identify the methods and source projects. Numerical rows in
this repository are our registered simulation results, **not** copied paper
tables or a claim of exact reproduction of each paper's highest-budget setup.

1. **Reducing the resources required by ADAPT-VQE using coupled exchange operators
   and improved subroutines.** [Published paper](https://www.nature.com/articles/s41534-025-01039-4),
   [author implementation](https://github.com/mafaldaramoa/ceo-adapt-vqe).
   The delivered local vendor bytes and Apache-2.0 notice identify the actual
   circuit implementation used; current upstream main is not substituted.
2. **Chemistry Beyond the Scale of Exact Diagonalization on a Quantum-Centric
   Supercomputer.** [Paper](https://arxiv.org/abs/2405.05068),
   [data repository](https://github.com/jrm874/sqd_data_repository),
   [archived data](https://doi.org/10.5281/zenodo.15324153).
   Context for sample-based quantum diagonalization and data organization.
3. **Improved parameter initialization for the (local) unitary cluster Jastrow
   ansatz.** [Paper v1](https://arxiv.org/abs/2511.22476v1),
   [source project](https://github.com/kevinsung/lucj).
   Captured C1 source metadata identifies commit
   `b52bca05650e60d890b06ac0a3b6d167303d0a90`; different registered LUCJ controls
   and initialization/readout conditions remain distinct.
4. **Quantum chemistry with provable convergence via randomized sample-based
   Krylov quantum diagonalization.** [SqDRIFT paper v3](https://arxiv.org/abs/2508.02578v3),
   [qiskit-fermions implementation project](https://github.com/Qiskit/qiskit-fermions).
   N=5/20/50 controls follow their own recorded SDK, randomization and budget
   settings; this reference is not a claim to duplicate all paper experiments.

Precise code/input lineage is in the original protocols, source manifests,
configuration inventory and protocol-binding audit. General repository-design
sources are listed separately in REPOSITORY_DESIGN.md.
