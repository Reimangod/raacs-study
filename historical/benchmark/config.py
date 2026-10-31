from dataclasses import dataclass, asdict

DISTANCES = {
    "H2": [0.7, 1.0, 1.5, 2.0, 3.0],
    "H4": [0.8, 1.1, 1.4, 1.8, 2.2],
    "H6": [0.8, 1.1, 1.4, 1.8, 2.2],
    "LiH": [1.0, 1.5, 2.0, 2.5, 3.0],
    "BeH2": [1.0, 1.3, 1.6, 2.0, 2.5],
    "N2": [0.9, 1.1, 1.4, 1.8, 2.2],
}


@dataclass(frozen=True)
class Case:
    molecule: str
    distance: float
    basis: str = "sto-3g"
    charge: int = 0
    spin: int = 0

    @property
    def id(self):
        return f"{self.molecule}_R{self.distance:.1f}"

    @property
    def frozen_core(self):
        return {"LiH": 1, "BeH2": 1, "N2": 2}.get(self.molecule, 0)

    def geometry(self):
        r = self.distance
        if self.molecule in ("H2", "H4", "H6"):
            return [("H", (0., 0., i * r)) for i in range(int(self.molecule[1:]))]
        if self.molecule == "LiH":
            return [("Li", (0., 0., 0.)), ("H", (0., 0., r))]
        if self.molecule == "BeH2":
            return [("H", (0., 0., -r)), ("Be", (0., 0., 0.)), ("H", (0., 0., r))]
        if self.molecule == "N2":
            return [("N", (0., 0., 0.)), ("N", (0., 0., r))]
        raise ValueError(self.molecule)

    def specification(self):
        return {**asdict(self), "geometry": self.geometry(), "frozen_core_spatial": self.frozen_core,
                "orbitals": "RHF canonical, ascending MO energy", "mapping": "Jordan-Wigner",
                "orbital_order": "interleaved alpha0,beta0,alpha1,beta1,...",
                "state_index": "OpenFermion big endian; logical q -> Qiskit wire n-1-q"}


def cases_for(mode):
    if mode == "quick":
        return [Case("H2", .7), Case("H4", 1.1)]
    names = ["H4", "H6", "LiH"] if mode == "standard" else list(DISTANCES)
    return [Case(name, r) for name in names for r in DISTANCES[name]]

