"""Bounded CPU comparator: paper-inspired HSQD remapping, common CI/OO."""
from importlib.metadata import version
from io import BytesIO
import os
from pathlib import Path
from time import process_time,perf_counter
import ffsim
import numpy as np
from qiskit import qpy
from frontierloop.half_remap import state,circuit,qualify
from frontierloop.marginal_remote_worker import sha,digest,load,save,write,npz,guard,resource_circuit,select_half_pool


def draw(root,case,row,spec):
    d=root/'outputs'/case;d.mkdir(parents=True,exist_ok=True);guard(root,spec)
    if (d/'draw-started.json').exists():raise RuntimeError('Partial H1 draw preserved, no automatic repetition')
    m=row['physical'];n=m['norb'];ne=m['nalpha'],m['nbeta']
    with np.load(root/row['operator_file'],allow_pickle=False) as a:
        source=ffsim.UCJOpSpinBalanced(a['diag_coulomb_mats'],a['orbital_rotations'],a['final_orbital_rotation'])
    cpu,tick=process_time(),perf_counter();prep,sampling,resources=resource_circuit(circuit(source,ne))
    qpy_hashes={}
    for kind,q in [('preparation',prep),('sampling',sampling)]:
        f=BytesIO();qpy.dump(q,f);payload=f.getvalue();write(d/(kind+'.qpy'),payload);qpy_hashes[kind]=sha(payload)
    build_cpu,build_wall=process_time()-cpu,perf_counter()-tick
    seed=int(digest([spec['seed_family'],m['problem_id']])[:8],16)
    save(d/'draw-started.json',{'run_id':row['run_id'],'model_shots':spec['model_shots_per_case'],'seed':seed})
    cpu,tick=process_time(),perf_counter();v=state(source,ne)
    if abs(float(np.vdot(v,v).real)-1)>1e-10:raise RuntimeError('H1 sector state norm failed')
    values=ffsim.sample_state_vector(v,norb=n,nelec=ne[0],shots=spec['model_shots_per_case'],bitstring_type=ffsim.BitstringType.INT,seed=seed)
    ids,counts=np.unique(values,return_counts=True)
    if int(counts.sum())!=spec['model_shots_per_case'] or any(int(x).bit_count()!=ne[0] for x in ids):raise RuntimeError('H1 counts/sector mismatch')
    payload=npz(half_strings=ids.astype(np.uint64),counts=counts.astype(np.uint64));write(d/'half-counts.npz',payload)
    result={'status':'HSQD_SPIN_REMAP_COMMON_SOURCE_DRAW_COMPLETED','run_id':row['run_id'],'physical_problem_id':m['problem_id'],
        'method':'HSQD_SPIN_REMAP_COMMON_SETUP_NOT_NATIVE_HCI','known_model_shots':int(counts.sum()),
        'half_configurations':int(counts.sum()),'seed':seed,'counts_SHA256':sha(payload),'resources':resources,'QPY_SHA256':qpy_hashes,
        'model_CNOT_applications':int(counts.sum())*resources['sampling']['cnot_count'],
        'compilation_CPU_seconds':build_cpu,'compilation_wall_seconds':build_wall,
        'sampling_CPU_seconds':process_time()-cpu,'sampling_wall_seconds':perf_counter()-tick,
        'nonredundant_parameter_rank':None,'original_marginal_identity_claimed':False,
        'source_initialization_record':row['source_initialization_record'],'actual_QPU_shots':0,'GPU_used':False,'statevector_saved':False}
    save(d/'draw-result.json',result);print('H1_DRAW',case,result['known_model_shots'],resources['sampling'],flush=True)
    return result


def readouts(root,case,row,spec):
    from frontierloop.orbital import optimize as ordinary
    from frontierloop.spin_orbital import optimize as penalized
    from frontierloop.fast_orbital import optimize as singlet
    d=root/'outputs'/case;r=load(d/'draw-result.json');m=row['physical'];ne=m['nalpha'],m['nbeta']
    if sha((d/'half-counts.npz').read_bytes())!=r['counts_SHA256']:raise RuntimeError('H1 draw binding drift')
    with np.load(d/'half-counts.npz',allow_pickle=False) as a:strings=select_half_pool(a['half_strings'],a['counts'],m['norb'],ne[0],spec['R_cap'])
    with np.load(root/row['integrals_file'],allow_pickle=False) as a:h,g=a['h1'],a['eri']
    for mode in spec['readouts']:
        guard(root,spec);out=d/mode;out.mkdir(exist_ok=True)
        if (out/'started.json').exists():raise RuntimeError('Existing H1 readout prefix, no automatic repetition')
        save(out/'started.json',{'counts_SHA256':r['counts_SHA256'],'CI_strings':strings.tolist()})
        cpu,tick=process_time(),perf_counter()
        try:
            settings=spec['readout_specs'][mode]
            def checkpoint(history,rotations):save(out/'checkpoint.json',{'history':history,'new_model_shots':0})
            if mode=='C4-SINGLET-OO':value=singlet(h,g,m['core_energy'],ne,strings,settings,mode,checkpoint)
            else:value=(ordinary if mode=='C2-OO' else penalized)(h,g,m['core_energy'],ne,strings,settings,checkpoint)
            rotations,hh,gg=(value.pop(k) for k in ('rotations','final_h1','final_eri'))
            payload=npz(rotations=np.array(rotations),h1=hh,eri=gg,CI_strings=strings);write(out/'optimized-representation.npz',payload)
            value.update(status='COMPLETED',case=case,method='HSQD_SPIN_REMAP_COMMON_SETUP_NOT_NATIVE_HCI',
                readout=mode,physical_problem_id=m['problem_id'],source_run_id=row['run_id'],counts_SHA256=r['counts_SHA256'],
                CI_strings=strings.tolist(),r_cap=spec['R_cap'],actual_CI_dimension=len(strings)**2,representation_SHA256=sha(payload),
                whole_body_CPU_seconds=process_time()-cpu,whole_body_wall_seconds=perf_counter()-tick,
                new_model_shots=0,actual_QPU_shots=0,CI_state_not_sampler_prepared_state=True,goal_complete=False)
            save(out/'result.json',value);print('H1_READOUT',case,mode,value['final_energy_Ha'],flush=True)
        except Exception as exc:
            save(out/'failure.json',{'status':'FAILED_ENGINEERING_PRESERVED','error':repr(exc),
                'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-tick,'new_model_shots':0})
            print('H1_READOUT_FAILED',case,mode,repr(exc),flush=True)


def main(root):
    root=Path(root).resolve();manifest=load(root/'manifest.json');spec=manifest['protocol']
    if (root/'started.json').exists():raise RuntimeError('Existing H1 job prefix, no automatic rerun')
    guard(root,spec)
    for name,expected in manifest['input_SHA256'].items():
        if sha((root/name).read_bytes())!=expected:raise RuntimeError('H1 uploaded input/code mismatch: '+name)
    runtime={p:version(p) for p in manifest['expected_runtime']}
    if runtime!=manifest['expected_runtime']:raise RuntimeError('Qualified package runtime changed')
    save(root/'started.json',{'job_id':os.environ.get('SLURM_JOB_ID'),'runtime':runtime,'GPU_used':False})
    save(root/'qualification.json',{'checks':qualify(),'new_model_shots':0})
    cpu,tick=process_time(),perf_counter();results=[]
    for case in spec['cases']:
        try:
            draw(root,case,manifest['cases'][case],spec);readouts(root,case,manifest['cases'][case],spec)
            results.append({'case':case,'status':'DRAW_AND_READOUT_ATTEMPTS_TERMINAL'})
        except Exception as exc:
            save(root/'outputs'/case/'failure.json',{'status':'FAILED_ENGINEERING_PRESERVED','error':repr(exc),'unknown_draws_not_refunded':True})
            results.append({'case':case,'status':'FAILED_ENGINEERING_PRESERVED'})
    save(root/'terminal.json',{'status':'H1_COMMON_SETUP_COMPARATOR_TERMINAL_NOT_SUPERIORITY','cases':results,
        'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-tick,'actual_QPU_shots':0,'GPU_used':False,'goal_complete':False})


if __name__=='__main__':
    import sys
    main(sys.argv[1])
