"""N48 dimension-only inventory: repeated observations vs SDK unique-weighted batches.

All saved 100k Fe source histograms are reused. No new quantum samples, eigenvalue
calculations, optimizer or CI result selection occurs in this module.
"""
from pathlib import Path
from time import process_time, perf_counter
import importlib.metadata
import inspect
import numpy as np
from qiskit_addon_sqd.subsampling import subsample
from qiskit_addon_sqd.fermion import batch_to_ci_strings
from .paths import ROOT, OLD
from .io import load, save, sha256, space

LOCAL=ROOT/'runs/N48-FE-READOUT-CONTRACT-v1'
CASE='FE2S2-40-P2'
METHODS=('C5_C_LUCJ_COMPRESSED_L1','C5_E_LUCJ_TN_COMMON_L1','C5_D_SQDRIFT_N5','C5_D_SQDRIFT_N20')
ARMS=('UNIFORM_OBSERVATION_POSITIONS_WITHOUT_REPLACEMENT','SDK_WEIGHTED_UNIQUE_WITHOUT_REPLACEMENT')


def validate(ids,counts,n,nelec,total):
    if ids.ndim!=1 or counts.ndim!=1 or len(ids)!=len(counts) or ids.dtype.kind not in 'iu' or counts.dtype.kind not in 'iu': raise ValueError('Integer histogram required')
    if len(set(map(int,ids)))!=len(ids) or np.any(counts<=0) or sum(map(int,counts))!=total: raise ValueError('Histogram totals or duplicates')
    if np.any(ids[1:]<=ids[:-1]): raise ValueError('Sorted unique input required')
    mask=(1<<n)-1
    if any(int(x)<0 or int(x)>=1<<(2*n) or (int(x)&mask).bit_count()!=nelec[0] or (int(x)>>n).bit_count()!=nelec[1] for x in ids): raise ValueError('Particle sector or register differs')


def occurrence_batch(ids,counts,size,rng):
    """Exactly the indices of uniform sampling of the unexpanded observations."""
    total=sum(map(int,counts))
    if size<1 or size>total: raise ValueError('Requested number of observations unavailable')
    positions=rng.choice(total,size=size,replace=False)
    return ids[np.searchsorted(np.cumsum(counts,dtype=np.int64),positions,side='right')]


def to_bits(ids,n):
    return ((np.asarray(ids,dtype=np.uint64)[:,None] >> np.arange(2*n-1,-1,-1,dtype=np.uint64))&1).astype(bool)


def dimensions(bits,n):
    a,b=batch_to_ci_strings(bits,n,symmetrize_spin=True)
    a,b=np.sort(a),np.sort(b)
    values=np.sum(bits.astype(np.uint64)<<np.arange(2*n-1,-1,-1,dtype=np.uint64),axis=1,dtype=np.uint64)
    mask=np.uint64((1<<n)-1)
    independent=np.union1d(values&mask,values>>np.uint64(n)).astype(np.int64)
    if not np.array_equal(a,independent) or not np.array_equal(a,b): raise ValueError('SDK blocked-bit / symmetric-space conversion differs')
    R=len(a);D=R*R
    # This is only vector-storage arithmetic, NOT a solver peak-RAM prediction.
    return dict(R_alpha=R,R_beta=R,D=D,one_real_vector_bytes=8*D,
                twenty_real_vectors_bytes=20*8*D,
                dense_H_bytes_if_materialized=8*D*D,
                half_strings=a.tolist(),full_observations_in_batch=len(bits),
                distinct_full_strings=len(np.unique(values)))


def run():
    if (LOCAL/'terminal.json').exists(): raise RuntimeError('Dimension inventory already completed; do not rerun')
    space(); cpu,wall=process_time(),perf_counter()
    source_manifest=ROOT/'runs/R2-COMMON-CI-v1/freeze.json'
    sources=[s for s in load(source_manifest)['sources'] if s['case']==CASE and s['method_id'] in METHODS]
    if len(sources)!=4: raise ValueError('All four specified source methods required')
    costs=load(ROOT/'analysis/SOURCE-cost-components-v1.json')
    physical=load(OLD/'data/physical'/CASE/'problem.json')
    rows=[]
    for si,s in enumerate(sources):
        if s['status']!='AVAILABLE' or s['expected_problem_id']!=physical['problem_id'] or s['source_final_model_shots']!=100000: raise ValueError('Physical/source identity differs')
        file=Path(s['payload']); result=Path(s['source_result'])
        if sha256(file.read_bytes())!=s['payload_SHA256'] or sha256(result.read_bytes())!=s['source_result_SHA256']: raise ValueError('Frozen original source changed')
        with np.load(file,allow_pickle=False) as z: ids,counts=z['blocked_ids'],z['counts']
        # Old histograms need not be in numeric ID order. Sort the in-memory
        # pair together, never the saved artifact or the counts alone.
        order=np.argsort(ids,kind='stable'); ids,counts=ids[order],counts[order]
        validate(ids,counts,20,(15,15),100000)
        bits=to_bits(ids,20);batches=[]
        for ai,arm in enumerate(ARMS):
            seed=480100+10*si+ai;rng=np.random.default_rng(seed)
            if ai==0:
                allbits=[to_bits(occurrence_batch(ids,counts,4000,rng),20) for _ in range(10)]
            else:
                allbits=subsample(bits,counts/counts.sum(),4000,10,rand_seed=rng)
            for j,b in enumerate(allbits):
                batches.append(dict(arm=arm,batch=j,seed=seed,**dimensions(b,20)))
        rows.append(dict(method=s['method_id'],source=s,cost=next(c for c in costs['rows'] if c['case']==CASE and c['method']==s['method_id']),
                         distinct_parent_full_strings=len(ids),parent_observation_count=int(counts.sum()),
                         batches=batches,statistically_independent_whole_runs=1))
    scipy_driver=LOCAL/'author-source/lucj_compressed_t2_task_sci.py'
    if not scipy_driver.exists(): raise ValueError('Pinned author source evidence absent')
    pinned_TN=OLD/'vendor/TN-source-v1/src/lucj/quimb_task/lucj_sqd_quimb_task_nomad.py'
    code=Path(__file__);sdk=Path(inspect.getfile(subsample));sdkfermion=Path(inspect.getfile(batch_to_ci_strings))
    record=dict(status='N48_ALL80_DIMENSION_ONLY_BATCHES_AUDITED_NO_NEW_ENERGIES',rows=rows,
          source_manifest_SHA256=sha256(source_manifest.read_bytes()),problem_id=physical['problem_id'],
          package_versions={p:importlib.metadata.version(p) for p in ['qiskit-addon-sqd','numpy','pyscf']},
          source_evidence=[dict(path=str(p),SHA256=sha256(p.read_bytes())) for p in [code,sdk,sdkfermion,scipy_driver,pinned_TN]],
          batch_counts=80,new_model_samples=0,new_QPU_shots=0,new_CI_roots=0,new_GPU_jobs=0,
          repeated_subsamples_not_new_quantum_work=True,independent_confirmation=False,
          paper_native_environment_exactly_identified=False,goal_complete=False,
          CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall)
    if record['CPU_seconds']>120 or record['wall_seconds']>180: raise RuntimeError('Preflight resource bounds exceeded')
    save(LOCAL/'terminal.json',record)
    print('N48_DIMENSIONS_NO_NEW_ENERGY')
    for row in rows:
        for arm in ARMS:
            z=[b for b in row['batches'] if b['arm']==arm]
            print(row['method'],arm,'R',min(b['R_alpha'] for b in z),max(b['R_alpha'] for b in z),
                  'D',min(b['D'] for b in z),max(b['D'] for b in z),flush=True)
    print('CPU',record['CPU_seconds'],'wall',record['wall_seconds'])


if __name__=='__main__': run()
