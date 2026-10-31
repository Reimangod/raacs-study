"""Resource-only order-preserving Rustiq successor exporter, not primary sampling.

Derived from sqdrift_native_worker.py. Same frozen grouping, seeds, relabeling
and Pauli order; only the exact synthesis backend changes. No extra draws.

This is an explicitly documented common-budget adaptation, not the unavailable
P1 author experiment driver. Original SDK algorithms are called, not patched.
Grouping coefficient refinement preserves H; fermionic relabeling preserves
the sampling problem and is decoded from actual circuit metadata.
"""
from collections import Counter
import gzip
from io import BytesIO
import json
from pathlib import Path
import sys
from time import perf_counter, process_time

import numpy as np

from .io import atomic_bytes, save_json, sha256


def span_permutation(excitations, n):
    """Bounded deterministic local solve of the SDK's multi-span objective.

    Not an assertion of the exact optimizer/optimum used in P1. No energy or
    measurement outcomes enter this calculation. Identity is a candidate so
    this routine never increases its explicitly registered objective.
    """
    supports = []
    for excitation in excitations:
        support = sorted(q for q, count in Counter(excitation).items() if count % 2)
        if len(support) in (2, 4) and support not in supports:
            supports.append(support)
    if not supports:
        return list(range(n)), {"initial": 0., "final": 0., "passes": 0}
    supports = np.array([s if len(s) == 4 else s+s for s in supports], dtype=int)

    def objective(permutation):
        spans = np.ptp(permutation[supports], axis=1)+1
        return float(np.max(spans)+.1*np.mean(spans))

    identity = np.arange(n)
    active = sorted(set(q for s in supports for q in s))
    order = active+[q for q in range(n) if q not in active]
    compact = np.argsort(order)
    current = compact if objective(compact) < objective(identity) else identity
    initial = objective(identity)
    passes = 0
    for _ in range(4):
        passes += 1
        best, best_score = current, objective(current)
        for a in active:
            for b in active:
                if a >= b:
                    continue
                trial = current.copy()
                trial[a], trial[b] = trial[b], trial[a]
                score = objective(trial)
                if score < best_score-1e-12:
                    best, best_score = trial, score
        if np.array_equal(best, current):
            break
        current = best
    return current.tolist(), {"objective": "SDK multi: max span +0.1 mean span", "initial": initial,
             "final": objective(current), "passes": passes, "global_optimality_claim": False,
             "paper_exact_layout_solver_claim": False}


class Exporter:
    def __init__(self, request):
        import qiskit
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_fermions.operators import FermionOperator
        from qiskit_fermions.operators.terms.ordering import canonical_order
        from qiskit_fermions.operators.terms.grouping import group_terms_by_electronic_structure
        if qiskit.__version__ != "2.5.0":
            raise RuntimeError("Use the isolated pinned SDK overlay")
        self.n = int(request["n_qubits"])
        self.hf = int(request["hf_id"])
        self.request = request
        tick, cpu = perf_counter(), process_time()
        terms = [(tuple((bool(create), int(q)) for q, create in row["term"]), complex(*row["coefficient"]))
                 for row in request["fermion_terms"]]
        operator = canonical_order(FermionOperator.from_terms(terms).normal_ordered().simplify(atol=1e-12))
        group_terms_by_electronic_structure(operator, self.n, two_body_physicist_order=False)
        groups = operator.split_out_groups()
        refined = []
        gid = 0
        self.raw_groups = len(groups)
        for group in groups:
            buckets = []
            for actions, coefficient in sorted(group.iter_terms(), key=lambda x: (abs(x[1]), repr(x[0]))):
                magnitude = abs(coefficient)
                if not buckets or not np.isclose(magnitude, buckets[-1][0], atol=1e-13, rtol=1e-11):
                    buckets.append((magnitude, []))
                buckets[-1][1].append((actions, coefficient))
            for _, items in buckets:
                refined.extend((actions, coefficient, gid) for actions, coefficient in items)
                gid += 1
        self.operator = FermionOperator.from_terms_with_groups(refined)
        self.weights = self.operator.group_weights()
        self.first_moment_error = max(abs(self.weights[group]*np.sign(coefficient)-coefficient)
                                      for _, coefficient, group in refined)
        if self.first_moment_error > 1e-10:
            raise RuntimeError("Group-weight normalization changes H")
        self.setup_wall = perf_counter()-tick
        self.setup_cpu = process_time()-cpu

    def export(self, spec, target):
        from qiskit import qpy
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_fermions.operators import FermionOperator
        from qiskit_fermions.mappers.library import jordan_wigner
        from qiskit_fermions.circuit import FermionicCircuit
        from qiskit_fermions.circuit.library import Evolution, InitializeModes
        from qiskit_fermions.transpiler import FermionicPassManager
        from qiskit_fermions.transpiler.converters import FermionicCircuitToDAG
        from qiskit_fermions.transpiler.passes import QDriftTrotterization, RelabelModes
        from qiskit_fermions.transpiler.presets import generate_preset_jw_pass_manager
        tick, cpu = perf_counter(), process_time()
        fc = FermionicCircuit(self.n)
        fc.append(InitializeModes([bool(self.hf >> (self.n-1-q) & 1) for q in range(self.n)]), fc.modes)
        fc.append(Evolution(self.n, self.operator, float(spec["time"])), fc.modes)
        randomized = FermionicPassManager([QDriftTrotterization(int(spec["num_terms"]), filter_trivial=False,
                                                       rng=int(spec["seed"]))]).run(fc)
        sequence = []
        excitations = []
        max_norm_bound = 0.
        for node in FermionicCircuitToDAG().run(randomized).topological_op_nodes():
            if not isinstance(node.op, Evolution):
                continue
            group = node.op.operator
            for actions, coefficient in group.iter_terms():
                if any(sum((1 if create else -1) for create, q in actions if q % 2 == spin) != 0 for spin in (0, 1)):
                    raise RuntimeError("Native excitation group breaks registered Nalpha/Nbeta")
            excitations.extend(tuple(q for _, q in actions) for actions, coefficient in group.iter_terms())
            pauli = SparsePauliOp.from_sparse_observable(jordan_wigner(group, self.n)).simplify(atol=1e-12)
            for i in range(len(pauli)):
                if abs(pauli.coeffs[i].imag) > 1e-10:
                    raise RuntimeError("Non-Hermitian sampled unit group")
                for j in range(i):
                    if not pauli.paulis[i].commutes(pauli.paulis[j]):
                        raise RuntimeError("Intra-group product is not exact; no silent first-order approximation")
            max_norm_bound = max(max_norm_bound, float(np.sum(abs(pauli.coeffs))))
            if max_norm_bound > 1+1e-9:
                raise RuntimeError("Unit group Pauli-L1 norm bound exceeds1")
            terms = [[label[::-1], float(coefficient.real*node.op.params[0])]
                     for label, coefficient in pauli.to_list() if set(label) != {"I"}]
            sequence.append(terms)
        if len(sequence) != spec["num_terms"]:
            raise RuntimeError("Native randomized excitation count mismatch")
        permutation, layout = span_permutation(excitations, self.n)
        relabeled = FermionicPassManager([RelabelModes(permutation=permutation)]).run(randomized)
        from qiskit import QuantumCircuit
        from qiskit.synthesis import synth_pauli_network_rustiq
        network = []
        for node in FermionicCircuitToDAG().run(relabeled).topological_op_nodes():
            if not isinstance(node.op,Evolution):
                continue
            # RelabelModes changes DAG wiring, not the local operator itself.
            # Match SDK evolution synthesis: relabel fermionic indices BEFORE JW.
            global_operator = node.op.operator.relabel_modes(permutation)
            pauli = SparsePauliOp.from_sparse_observable(jordan_wigner(global_operator,self.n)).simplify(atol=1e-12)
            for label,coefficient in pauli.to_list():
                if abs(coefficient.imag)>1e-10: raise RuntimeError("Non-Hermitian network")
                support=[q for q,p in enumerate(label[::-1]) if p!="I"]
                if support:
                    word="".join(label[::-1][q] for q in support)
                    network.append((word,support,float(2*coefficient.real*node.op.params[0])))
        circuit=QuantumCircuit(self.n)
        for old in range(self.n):
            if self.hf >> (self.n-1-old) & 1:
                circuit.x(permutation[old])
        circuit.compose(synth_pauli_network_rustiq(self.n,network,optimize_count=True,
            preserve_order=True,upto_clifford=False,upto_phase=False),inplace=True)
        circuit.metadata={"permutation":permutation,"synthesis":"order-preserving Qiskit bundled Rustiq"}
        actual_permutation = (circuit.metadata or {}).get("permutation")
        if actual_permutation != permutation:
            raise RuntimeError("Actual native F2Q relabeling metadata missing/different")
        out = BytesIO()
        qpy.dump(circuit, out)
        atomic_bytes(target/"circuit.qpy.gz", gzip.compress(out.getvalue(), mtime=0))
        report = {"status": "NATIVE_GROUPED_MEMBER_EXPORTED", "spec": spec, "permutation": actual_permutation,
                  "source_mode_group_Pauli_sequence": sequence, "native_F2Q_layout": layout,
                  "max_unit_group_Pauli_L1_norm_bound": max_norm_bound,
                  "first_moment_error": self.first_moment_error,
                  "runtime": {"qiskit": "2.5.0", "qiskit_fermions": "0.1.0"},
                  "partition": "SDK electronic symmetry groups, coefficient-magnitude refinement",
                  "diagonal_terms_retained": True, "trivial_excitation_filter": False,
                  "paper_author_driver_reproduced": False,
                  "resource_only_successor_no_new_sampling": True, "synthesis": "bundled Qiskit2.5 Rustiq, exact Pauli order",
                  "adaptation": "Common-budget native SDK; bounded deterministic F2Q multi-span local optimization",
                  "export_wall_seconds": perf_counter()-tick, "export_CPU_seconds": process_time()-cpu,
                  "circuit_file_sha256": sha256((target/"circuit.qpy.gz").read_bytes())}
        payload = (json.dumps(report, sort_keys=True)+"\n").encode()
        atomic_bytes(target/"member.json.gz", gzip.compress(payload, mtime=0))
        return {"status": report["status"], "export_wall_seconds": report["export_wall_seconds"],
                "member_sha256": sha256(payload)}


def serve(base, directory):
    directory = Path(directory).resolve()
    exporter = Exporter(json.loads(Path(base).read_text()))
    number = len(list((directory/"group-preparations").glob("*.json")))+1
    save_json(directory/"group-preparations"/(str(number)+".json"), {"raw_group_count": exporter.raw_groups,
              "refined_group_count": len(exporter.weights), "first_moment_error": exporter.first_moment_error,
              "wall_seconds": exporter.setup_wall, "CPU_seconds": exporter.setup_cpu,
              "worker_sha256": sha256(Path(__file__).read_bytes()), "model_quantum_shots": 0})
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("stop"):
            return
        target = (directory/request["relative_directory"]).resolve()
        if directory not in target.parents:
            raise RuntimeError("Output outside declared member root")
        target.mkdir(parents=True, exist_ok=True)
        try:
            result = exporter.export(request["spec"], target)
        except Exception as exc:
            import traceback
            result = {"status": "FAILED_PRESERVED", "error": repr(exc), "traceback": traceback.format_exc()}
        save_json(target/"export-event.json", result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    serve(args.base, args.directory)
