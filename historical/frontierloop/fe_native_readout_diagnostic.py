"""N47 fixed saved Fe spaces: SDK physical-H vs prior H+.1S2 readout.

Retrospective diagnostic, not fresh finite-shot confirmation or paper reproduction.
No source, sample, CI basis, or historical result is modified.
"""
from pathlib import Path
from time import process_time, perf_counter
import inspect
import sys
import numpy as np
from scipy.sparse.linalg import eigsh
from qiskit_addon_sqd.fermion import solve_sci
from .paths import ROOT, OLD
from .io import load, save, sha256, space
from .joint_ci_sparse import matrices

LOCAL=ROOT/'runs/N47-FE-NATIVE-READOUT-DIAGNOSTIC-v2'
SUMMARY=ROOT/'analysis/N45-ALL-summary.json'
METHODS=('FIXED_U_LINEAR_J_FIT_L1','C5_C_LUCJ_COMPRESSED_L1')
LIBRARY=ROOT/'runs/N17-M4-CI-WITNESS-v1/library/libjoint_ci.dylib'


def sdk_result(h,g,core,n,ne,strings,library):
    cpu,wall=process_time(),perf_counter()
    strings=np.array(sorted(strings),dtype=np.int64)
    pairs=np.array([(int(a),int(b)) for a in strings for b in strings],dtype=np.int64)
    z=solve_sci((strings,strings),h,g,n,ne,spin_sq=None,
                tol=1e-12,tol_residual=1e-9,lindep=1e-14,max_cycle=100,max_memory=512,verbose=0)
    state=z.sci_state
    if not np.array_equal(state.ci_strs_a,strings) or not np.array_equal(state.ci_strs_b,strings):
        raise ValueError('SDK determinant order changed')
    v=np.asarray(state.amplitudes).ravel()
    H,S,info=matrices(h,g,ne,pairs,library,seconds=10,triangle_cap=600000)
    energy=float(v@(H@v)+core); spin=float(v@(S@v))
    norm=abs(float(v@v)-1); residual=float(np.linalg.norm(H@v-(energy-core)*v))
    rng=np.random.default_rng(4701)
    ev,vec=eigsh(H,k=1,which='SA',tol=1e-11,maxiter=2000,v0=rng.normal(size=len(v)))
    independent_error=abs(energy-core-float(ev[0]))
    if max(norm,residual,independent_error,abs(energy-float(z.energy)-core))>1e-7:
        raise ValueError('SDK RDM, sparse physical-H, norm/residual, or independent lowest root mismatch')
    return dict(energy_Ha=energy,spin_square=spin,SDK_RDM_energy_Ha=float(z.energy)+core,
                residual_norm=residual,norm_error=norm,independent_lowest_root_difference_Ha=independent_error,
                actual_R=len(strings),actual_D=len(pairs),SDK_eigensolves=1,independent_audit_eigensolves=1,
                CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,**info)


def prepare():
    if (LOCAL/'protocol.json').exists(): raise RuntimeError('Protocol already exists; inspect instead of overwrite')
    space(); a=load(SUMMARY)
    rows=[x for x in a['rows'] if x['case']=='FE2S2-40-P2' and x['arm']=='R32_CAP' and x['method'] in METHODS]
    if len(rows)!=6 or any(x['status']!='COMPLETED' or x['R']!=32 for x in rows): raise ValueError('Expected six fixed R32 spaces')
    path=OLD/'data/physical/FE2S2-40-P2/integrals.npz'
    sdk=Path(inspect.getfile(solve_sci))
    p=dict(scope='POSTHOC_SAME_SAVED_SPACES_DIAGNOSTIC_NOT_INDEPENDENT_CONFIRMATION',
           methods=list(METHODS),rows=rows,summary_SHA256=sha256(SUMMARY.read_bytes()),
           integrals=str(path),integrals_SHA256=sha256(path.read_bytes()),
           library=str(LIBRARY),library_SHA256=sha256(LIBRARY.read_bytes()),
           SDK_source=str(sdk),SDK_SHA256=sha256(sdk.read_bytes()),
           executor_SHA256=sha256(Path(__file__).read_bytes()),
           equation='PROJECTED_PHYSICAL_H_NO_SPIN_PENALTY_NO_RECOVERY',
           new_model_samples=0,new_QPU_shots=0,new_GPU_jobs=0,
           no_parameter_or_basis_or_source_selection=True,
           per_cell_CPU_wall_seconds=[10,30],new_storage_cap_bytes=4*2**20)
    save(LOCAL/'protocol.json',p); print('N47_PREPARED_SIX_FIXED_SPACES')


def run():
    if (LOCAL/'terminal.json').exists(): raise RuntimeError('Completed diagnostic exists; do not repeat')
    space(); p=load(LOCAL/'protocol.json')
    for path,key in [(p['integrals'],'integrals_SHA256'),(p['library'],'library_SHA256'),
                     (p['SDK_source'],'SDK_SHA256'),(__file__,'executor_SHA256')]:
        if sha256(Path(path).read_bytes())!=p[key]: raise ValueError('Frozen execution input differs')
    # The exact basis comes from each saved root, not re-ranking or resampling.
    with np.load(p['integrals'],allow_pickle=False) as a: h,g=a['h1'],a['eri']
    cpu,wall=process_time(),perf_counter(); rows=[]
    for i,old in enumerate(p['rows']):
        file=LOCAL/f'cell-{i}.json'
        if file.exists():
            row=load(file)
            if row['protocol_SHA256']!=sha256((LOCAL/'protocol.json').read_bytes()): raise ValueError('Existing cell context differs')
            rows.append(row); continue
        source=Path(old['root_path'])
        if sha256(source.read_bytes())!=old['root_SHA256']: raise ValueError('Saved basis changed')
        with np.load(source,allow_pickle=False) as z: pairs=z['joint_pairs']
        strings=sorted({int(x) for x in pairs[:,0]})
        if pairs.tolist()!=[[a,b] for a in strings for b in strings]: raise ValueError('Not the fixed symmetric Cartesian basis')
        result=sdk_result(h,g,0.,20,(15,15),strings,p['library'])
        if result['CPU_seconds']>10 or result['wall_seconds']>30: raise RuntimeError('Cell timing bound exceeded')
        row=dict(case=old['case'],method=old['method'],replicate=old['replicate'],
                 old_energy_Ha=old['energy_Ha'],old_spin_square=old['spin_square'],basis_id=old['basis_id'],
                 source_root_SHA256=old['root_SHA256'],result=result,
                 protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()))
        save(file,row); rows.append(row); print('N47_CELL',i,old['method'],result['energy_Ha'],flush=True)
    means={method:float(np.mean([x['result']['energy_Ha'] for x in rows if x['method']==method])) for method in METHODS}
    oldmeans={method:float(np.mean([x['old_energy_Ha'] for x in rows if x['method']==method])) for method in METHODS}
    out=dict(status='N47_SIX_PHYSICAL_H_SDK_READOUTS_INDEPENDENTLY_AUDITED',rows=rows,
             energy_mean_Ha=means,old_penalized_root_physical_energy_mean_Ha=oldmeans,
             fitted_minus_CDF_mHa=1000*(means[METHODS[0]]-means[METHODS[1]]),
             new_model_samples=0,new_QPU_shots=0,new_GPU_jobs=0,
             SDK_eigensolves=6,independent_audit_eigensolves=6,
             independent_confirmation=False,native_paper_full_readout_reproduced=False,goal_complete=False,
             CPU_seconds_this_invocation=process_time()-cpu,wall_seconds_this_invocation=perf_counter()-wall,
             cell_CPU_seconds_total=sum(x['result']['CPU_seconds'] for x in rows),
             protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()))
    save(LOCAL/'terminal.json',out); print('N47_TERMINAL_DELTA_mHa',out['fitted_minus_CDF_mHa'])


if __name__=='__main__': {'prepare':prepare,'run':run}[sys.argv[1]]()
