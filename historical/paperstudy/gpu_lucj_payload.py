"""Serialize exact source LUCJ primitives; no new ansatz fitting."""
import numpy as np
import ffsim

from .io import digest


def payload(op, norb, nelec, requests=None):
    rotations = []
    current = np.eye(norb)
    matrices = []
    for U in op.orbital_rotations:
        matrices.append(U.conj().T@current)
        current = U
    matrices.append(current if op.final_orbital_rotation is None else op.final_orbital_rotation@current)
    for U in matrices:
        givens, diagonal = ffsim.linalg.givens_decomposition(U)
        rotations.append({"givens": [[float(c), float(s.real), float(s.imag), int(i), int(j)] for c, s, i, j in givens],
                          "diagonal": [[float(x.real), float(x.imag)] for x in diagonal]})
    config = {"norb": norb, "nelec": list(nelec), "rotations": rotations,
              "source": "Exact uncompressed ffsim.linalg.givens_decomposition and UCJOpSpinBalanced basis ordering",
              "givens_compression_used": False, "sampling_requests": requests or []}
    config["input_id"] = digest(config)
    return config, {"diag_coulomb_mats": op.diag_coulomb_mats}
