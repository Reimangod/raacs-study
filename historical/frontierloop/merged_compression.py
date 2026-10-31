"""Native generic compression AFTER exact PRE_INIT circuit optimization.

This changes compiler order, not a new sampling objective. Materialize both
Slater preparation and orbital rotations so the simulator cannot silently
apply an uncompressed operator. Source datasets are never modified.
"""
import fcntl
import inspect
from io import BytesIO
from time import process_time, perf_counter
import numpy as np
import ffsim
from qiskit import QuantumCircuit, qpy
from .paths import ROOT, OLD
from .io import load, save, write, sha256, space
from .native_compression import OPTIONS, provenance as prior_provenance, single_particle_matrix
from paperstudy.lucj import make_circuit, circuit_views


def materialize(op, nelec, max_layers=None):
    exact = ffsim.qiskit.PRE_INIT.run(make_circuit(op, op.norb, nelec))
    result = exact.copy_empty_like(); diagnostics = []
    kwargs = OPTIONS if max_layers is not None else {}
    for item in exact:
        old = item.operation
        if isinstance(old, ffsim.qiskit.PrepareSlaterDeterminantJW):
            new = ffsim.qiskit.PrepareSlaterDeterminantJW(op.norb,
                old.occupied_orbitals, (old.orbital_rotation_a, old.orbital_rotation_b),
                max_layers=max_layers, **kwargs)
            diagnostics.append((old, new.definition.copy()))
        elif isinstance(old, ffsim.qiskit.OrbitalRotationJW):
            new = ffsim.qiskit.OrbitalRotationJW(op.norb,
                (old.orbital_rotation_a, old.orbital_rotation_b),
                max_layers=max_layers, **kwargs)
            diagnostics.append((old, new.definition.copy()))
        else:
            new = old
        result.append(new, [result.qubits[exact.find_bit(q).index] for q in item.qubits])
    result = result.decompose(gates_to_decompose=['slater_jw', 'orb_rot_jw'])
    if any(isinstance(i.operation, (ffsim.qiskit.PrepareSlaterDeterminantJW,
            ffsim.qiskit.OrbitalRotationJW, ffsim.qiskit.UCJOpSpinBalancedJW)) for i in result):
        raise RuntimeError('High-level approximation bypass is forbidden')
    return result, diagnostics


def prep_infidelity(old, circuit):
    """Factorized spin-sector Slater check, not full correlated-state fidelity."""
    n = old.norb; overlap_squared = 1.
    for spin, occupied in enumerate(old.occupied_orbitals):
        q = QuantumCircuit(n)
        for item in circuit:
            wires = [circuit.find_bit(w).index for w in item.qubits]
            if not wires:
                continue  # global phase does not affect infidelity
            if all(spin*n <= w < (spin+1)*n for w in wires):
                q.append(item.operation, [w-spin*n for w in wires])
            elif any(spin*n <= w < (spin+1)*n for w in wires):
                raise RuntimeError('Slater circuit unexpectedly couples spins')
        actual = ffsim.qiskit.final_state_vector(q, norb=n, nelec=len(occupied)).vec
        target = ffsim.slater_determinant(n, occupied,
            orbital_rotation=old.orbital_rotation_a if spin==0 else old.orbital_rotation_b)
        overlap_squared *= abs(np.vdot(target, actual))**2
    return float(max(0., 1-overlap_squared))


def binding():
    import ffsim.linalg.givens as givens
    import ffsim.qiskit.gates.slater_determinant as slater
    sources = {}
    for case in load(ROOT/'protocol-D1-MERGED-RESOURCE.json')['cases']:
        d = OLD/'prepared'/case/'C_LUCJ_COMPRESSED_L1-L1'
        sources[case] = {str(p):sha256(p.read_bytes()) for p in (
            d/'operator.npz', d/'construction.json', OLD/'data/physical'/case/'problem.json')}
    return {'adapter_SHA256':sha256((ROOT/'frontierloop/merged_compression.py').read_bytes()),
        'protocol_SHA256':sha256((ROOT/'protocol-D1-MERGED-RESOURCE.json').read_bytes()),
        'native_runtime':prior_provenance(),
        'slater_compiler_SHA256':sha256(inspect.getsource(slater).encode()),
        'givens_runtime_SHA256':sha256(inspect.getsource(givens).encode()),
        'sources':sources, 'qualification_SHA256':sha256((ROOT/'qualification-merged-compression-tests.xml').read_bytes())}


def run():
    spec=load(ROOT/'protocol-D1-MERGED-RESOURCE.json')
    if load(ROOT/'freeze-D1-MERGED-RESOURCE.json')!=binding():
        raise RuntimeError('D1 frozen input/runtime drift')
    directory=ROOT/'runs/D1-MERGED-RESOURCE';directory.mkdir(parents=True,exist_ok=True)
    with (directory/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (directory/'result.json').exists(): return load(directory/'result.json')
        if (directory/'started.json').exists(): raise RuntimeError('Preserve partial D1; no blind rerun')
        save(directory/'started.json',{'freeze':binding()})
        rows=[]
        for case in spec['cases']:
            source=OLD/'prepared'/case/'C_LUCJ_COMPRESSED_L1-L1'
            m=load(OLD/'data/physical'/case/'problem.json')
            with np.load(source/'operator.npz',allow_pickle=False) as a:
                op=ffsim.UCJOpSpinBalanced(a['diag_coulomb_mats'],a['orbital_rotations'],a['final_orbital_rotation'])
            nelec=m['nalpha'],m['nbeta']
            _,_,baseline=circuit_views(make_circuit(op,op.norb,nelec))
            for fraction in spec['layer_fractions']:
                space();cap=max(1,int(np.ceil(fraction*op.norb)))
                cpu,tick=process_time(),perf_counter();qc,diagnostics=materialize(op,nelec,cap)
                errors=[]
                for old,primitive in diagnostics:
                    if isinstance(old,ffsim.qiskit.PrepareSlaterDeterminantJW):
                        errors.append({'kind':'factorized_initial_Slater_state',
                            'infidelity':prep_infidelity(old,primitive)})
                    else:
                        effective=single_particle_matrix(primitive,op.norb)
                        errors.append({'kind':'single_particle_orbital_unitary',
                            'infidelity':float(max(0,1-abs(np.vdot(old.orbital_rotation_a,effective)/op.norb)**2))})
                prep,sampling,resources=circuit_views(qc)
                d=directory/case/str(cap);d.mkdir(parents=True,exist_ok=True);hashes={}
                for name,c in [('native-materialized',qc),('preparation',prep),('sampling',sampling)]:
                    stream=BytesIO();qpy.dump(c,stream);payload=stream.getvalue()
                    write(d/(name+'.qpy'),payload);hashes[name]=sha256(payload)
                row={'case':case,'problem_id':m['problem_id'],'fraction':fraction,'max_layers':cap,
                    'resources':resources,'exact_optimized_source_resources':baseline,
                    'part_errors_not_final_correlated_state_fidelity':errors,
                    'compiled_numeric_coefficient_slots_not_rank':sum(len(i.operation.params) for i in qc),
                    'nonredundant_independent_parameters':None,
                    'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-tick,
                    'QPY_SHA256':hashes,'new_model_shots':0,'actual_QPU_shots':0,
                    'energy_Ha':None,'recovery_pct':None,'full_correlated_state_fidelity':None,
                    'native_generic_compression_not_claimed_new':True}
                save(d/'result.json',row);rows.append(row)
                print('D1',case,cap,resources['Z_sampling_equivalent'],errors,flush=True)
        result={'status':'12_OPTIMIZATION_FIRST_RESOURCE_POINTS_TERMINAL_NOT_PERFORMANCE',
            'freeze':binding(),'rows':rows,'new_model_shots':0,'goal_complete':False}
        save(directory/'result.json',result);return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    if a.freeze:
        import xml.etree.ElementTree as ET
        suites=list(ET.parse(ROOT/'qualification-merged-compression-tests.xml').getroot().iter('testsuite'))
        if sum(int(s.get('tests','0')) for s in suites)!=5 or any(int(s.get(k,'0')) for s in suites for k in ('failures','errors')):
            raise RuntimeError('D1 state/resource qualification missing')
        path=ROOT/'freeze-D1-MERGED-RESOURCE.json';value=binding()
        if path.exists() and load(path)!=value: raise RuntimeError('Never replace D1 freeze')
        save(path,value)
    else:run()
