"""Isolated Qiskit2.5/fermions0.1 exporter, called only via its overlay process.

Uses licensed official SDK APIs, not copied implementation. No FCI, no global
diagonal filtering, no statevector. Exports actual randomized fermion circuit
and its sampled commuting Pauli factors for independent parity in main runtime.
"""
from __future__ import annotations
import argparse
from io import BytesIO
import json
from pathlib import Path
import numpy as np


def export(request, output):
    import qiskit
    from qiskit import qpy
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_fermions.operators import FermionOperator
    from qiskit_fermions.operators.terms.ordering import canonical_order
    from qiskit_fermions.operators.terms.grouping import group_terms_by_electronic_structure
    from qiskit_fermions.mappers.library import jordan_wigner
    from qiskit_fermions.circuit import FermionicCircuit
    from qiskit_fermions.circuit.library import InitializeModes, Evolution
    from qiskit_fermions.transpiler import FermionicPassManager
    from qiskit_fermions.transpiler.converters import FermionicCircuitToDAG
    from qiskit_fermions.transpiler.passes import QDriftTrotterization
    from qiskit_fermions.transpiler.presets import generate_preset_jw_pass_manager
    if qiskit.__version__ != "2.5.0":
        raise RuntimeError("Run SDK exporter in its isolated pinned overlay")
    output = Path(output); output.mkdir(exist_ok=True, parents=True)
    n = request["n_qubits"]
    # Explicit mode indices, interleaved alpha/beta. The outer adapter reverses
    # qubit wire labels only, not fermionic mode ordering.
    terms = [(tuple((bool(creation), int(q)) for q, creation in row["term"]), complex(*row["coefficient"])) for row in request["fermion_terms"]]
    operator = FermionOperator.from_terms(terms).normal_ordered().simplify(atol=1e-12)
    operator = canonical_order(operator)
    group_terms_by_electronic_structure(operator, n, two_body_physicist_order=False)
    weights = operator.group_weights(); groups = operator.split_out_groups()
    if weights is None or groups is None:
        raise RuntimeError("Electronic structure grouping missing")
    original_group_count = len(groups)
    original_max_first_moment_error = max((abs(w*np.sign(c)-c) for w, g in zip(weights, groups) for _, c in g.iter_terms()), default=0.)
    # Normal ordering can combine unequal Coulomb/exchange magnitudes. A unit-
    # sign group weighted by their mean would change H. Split by magnitude,
    # preserving Hermitian partners, with no measurement/outcome information.
    refined = []; group_id = 0
    for group in groups:
        buckets = []
        for actions, coefficient in sorted(group.iter_terms(), key=lambda v: (abs(v[1]), repr(v[0]))):
            magnitude = abs(coefficient)
            if not buckets or not np.isclose(magnitude, buckets[-1][0], atol=1e-13, rtol=1e-11):
                buckets.append((magnitude, []))
            buckets[-1][1].append((actions, coefficient))
        for _, members in buckets:
            refined.extend((a, c, group_id) for a, c in members); group_id += 1
    operator = FermionOperator.from_terms_with_groups(refined)
    weights = operator.group_weights(); groups = operator.split_out_groups()
    max_group_l1 = 0.; max_first_moment_error = 0.
    for weight, group in zip(weights, groups):
        for actions, coefficient in group.iter_terms():
            # Mean-magnitude grouping is valid only when magnitudes agree.
            max_first_moment_error = max(max_first_moment_error, abs(weight*np.sign(coefficient)-coefficient))
        unit = FermionOperator.from_terms([(actions, np.sign(coefficient)) for actions, coefficient in group.iter_terms()])
        pauli = SparsePauliOp.from_sparse_observable(jordan_wigner(unit, n)).simplify(atol=1e-12)
        if np.max(abs(pauli.coeffs.imag)) > 1e-10:
            raise RuntimeError("Magnitude refinement did not retain Hermitian group")
        max_group_l1 = max(max_group_l1, float(sum(abs(pauli.coeffs))))
    if max_first_moment_error > 1e-10:
        raise RuntimeError("Grouped SDK weights do not reconstruct this Hamiltonian")
    # A norm upper bound, not a claim that all groups saturate unit norm.
    if max_group_l1 > 1+1e-9:
        raise RuntimeError("Unit-group Pauli-L1 bound exceeds1; needs explicit norm treatment")
    fc = FermionicCircuit(n)
    occupation = np.array([bool(request["hf_id"] >> (n-1-q) & 1) for q in range(n)])
    fc.append(InitializeModes(occupation), range(n))
    fc.append(Evolution(n, operator, time=request["time"]), range(n))
    randomize = FermionicPassManager([QDriftTrotterization(request["num_terms"], filter_trivial=False, rng=int(request["seed"]))])
    randomized = randomize.run(fc)
    sequence = []; groups_sampled = []
    randomized_dag = FermionicCircuitToDAG().run(randomized)
    for node in randomized_dag.topological_op_nodes():
        if isinstance(node.op, Evolution):
            pauli = SparsePauliOp.from_sparse_observable(jordan_wigner(node.op.operator, n)).simplify(atol=1e-12)
            for i in range(len(pauli)):
                for j in range(i):
                    if not pauli.paulis[i].commutes(pauli.paulis[j]):
                        raise RuntimeError("Noncommuting intra-group synthesis requires separate qualification")
            delta = float(node.op.params[0])
            group_record = []
            for label, coefficient in pauli.to_list():
                if abs(coefficient.imag) > 1e-10:
                    raise RuntimeError("Non-Hermitian randomized group")
                if set(label) == {"I"}:
                    continue
                # Reverse Qiskit little-endian labels to the source raw index.
                sequence.append([label[::-1], float(coefficient.real*delta)])
                group_record.append([label[::-1], float(coefficient.real)])
            groups_sampled.append({"time_delta": delta, "pauli_terms": group_record})
    if len(groups_sampled) != request["num_terms"]:
        raise RuntimeError("Randomized group count differs from frozen request")
    pm = generate_preset_jw_pass_manager(basis_gates=["rz", "ry", "rx", "h", "s", "sdg", "x", "cx"], optimization_level=0, seed_transpiler=8137)
    qubit = pm.run(randomized)
    stream = BytesIO(); qpy.dump(qubit, stream)
    (output / "sdk-circuit.qpy").write_bytes(stream.getvalue())
    result = {"qiskit": qiskit.__version__, "qiskit_fermions": "0.1.0", "n_qubits": n,
              "partition": "SDK electronic groups refined by coefficient magnitude after normal ordering",
              "raw_SDK_group_count": original_group_count, "raw_SDK_first_moment_mismatch_before_refinement": original_max_first_moment_error,
              "group_count": len(groups), "randomized_group_count": len(groups_sampled),
              "group_weights": weights, "max_group_first_moment_error": max_first_moment_error,
              "max_unit_group_Pauli_L1_norm_upper_bound": max_group_l1,
              "full_Hamiltonian_diagonal_terms_retained": True, "filter_trivial": False,
              "omitted_terms": "scalar identity only, omitted by outer input adapter",
              "sequence_for_independent_source_basis_parity": sequence, "sampled_groups": groups_sampled,
              "source_wire_order_requires_reversal": True, "hardware_shots": 0,
              "scope": "one actual randomized circuit; full SqDRIFT requires multi-time multi-randomization sampling"}
    (output / "sdk-result.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    a = p.parse_args(); result = export(json.loads(Path(a.input).read_text()), a.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ("sequence_for_independent_source_basis_parity", "sampled_groups", "group_weights")}))
