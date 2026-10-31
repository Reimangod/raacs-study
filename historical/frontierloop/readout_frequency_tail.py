"""Additive completion of controls blocked by an unrelated screening timeout.

Never retries Global/native screens or changes a T3 result, basis, or time cap.
This is post-hoc operational completion, not a new confirmation experiment.
"""
import json,sqlite3,traceback
from pathlib import Path
from time import process_time,perf_counter
import numpy as np
from .paths import ROOT,OLD
from .io import load,save,sha256,digest,space
from .readout_seed_screen import RUN as T3,FREEZE,binding
from .readout_seed_plan import evaluation_key

RUN=ROOT/'runs/T3-FREQUENCY-TAIL-v1'


def missing_frequency_plans(plans,records):
    if len({r['cell_id'] for r in records})!=len(records):
        raise RuntimeError('Duplicate source-cell result')
    by_id={r['cell_id']:r for r in records}
    if set(by_id)!={p['cell_id'] for p in plans}:raise RuntimeError('Incomplete terminal coverage')
    return [p for p in plans if not any(r['arm']=='SAME_UNION_FREQUENCY' and
        r['status']=='COMPLETED' for r in by_id[p['cell_id']].get('rows',[]))]


def run():
    frozen=load(FREEZE);terminal=load(T3/'terminal.json');catalog=T3/'evaluations.sqlite'
    if binding()!=frozen or sha256(catalog.read_bytes())!=terminal['catalog_SHA256']:
        raise RuntimeError('Frozen T3 inputs or terminal catalog changed')
    records=[]
    for p in frozen['plans']:
        folder=T3/'cells'/p['cell_id']
        records.append(load(folder/('result.json' if (folder/'result.json').exists() else 'failure.json')))
    plans=missing_frequency_plans(frozen['plans'],records);space()
    settings=load(ROOT/'protocol-T3-READOUT-SEEDS.json')
    protocol=dict(T3_freeze_SHA256=sha256(FREEZE.read_bytes()),
        T3_terminal_SHA256=sha256((T3/'terminal.json').read_bytes()),
        T3_catalog_SHA256=terminal['catalog_SHA256'],implementation_SHA256=sha256(Path(__file__).read_bytes()),
        plans=plans,per_CI_wall_cap_seconds=settings['per_CI_wall_cap_seconds'],
        scope='ONLY_MISSING_FIXED_FREQUENCY_CONTROLS_NO_SCREEN_RETRY_POSTHOC_OPERATIONAL_COMPLETION',
        original_T3_results_immutable=True,new_model_shots=0,new_QPU_shots=0)
    RUN.mkdir(exist_ok=False);save(RUN/'protocol-freeze.json',protocol)
    db=sqlite3.connect('file:'+str(catalog)+'?mode=ro',uri=True)
    cpu,wall=process_time(),perf_counter();out=[]
    from .trim_readout import Solver
    try:
        for p in plans:
            space();start,tick=process_time(),perf_counter();source=p['source'];ctx=p['context']
            row=dict(cell_id=p['cell_id'],case=source['case'],method=source['method_id'],seed=p['request']['seed'],
                arm='SAME_UNION_FREQUENCY',original_T3_status=next(r['status'] for r in records if r['cell_id']==p['cell_id']),
                new_model_shots=0,new_QPU_shots=0,Global_or_native_screen_retried=False)
            try:
                strings=p['request']['same_union_frequency'];key=evaluation_key(ctx,strings)
                found=db.execute('SELECT id,origin,record,cpu FROM observations WHERE lookup=? ORDER BY seq LIMIT 1',(key,)).fetchone()
                if found:
                    oid,origin,record,proxy=found;root=json.loads(record);new=0
                else:
                    physical=OLD/'data/physical'/source['case'];meta=load(physical/'problem.json')
                    with np.load(physical/'integrals.npz',allow_pickle=False) as a:h,g=a['h1'],a['eri']
                    solver=Solver(h,g,meta['core_energy'],tuple(ctx['nelec']),dict(r_cap=128,spin_penalty_Ha=.1,
                        per_CI_wall_cap_seconds=settings['per_CI_wall_cap_seconds'],
                        total_wall_cap_seconds=settings['per_cell_wall_cap_seconds'],
                        total_CPU_cap_seconds=settings['per_cell_CPU_cap_seconds']))
                    ss=np.asarray(strings,np.int64);solver([(ss,ss)],h,g,ctx['norb'],tuple(ctx['nelec']))
                    root=solver.calls[-1];proxy=root['CPU_seconds'];new=1
                    origin=str(RUN/p['cell_id']/'root.json');oid=digest([ctx,key,origin,root])
                if root['CI_strings']!=strings or abs(root['energy_Ha']+.1*root['spin_square']-root['penalized_energy_Ha'])>1e-8:
                    raise RuntimeError('Frequency root basis or spin identity mismatch')
                save(RUN/p['cell_id']/'root.json',root)
                row.update(status='COMPLETED',energy_Ha=root['energy_Ha'],S2=root['spin_square'],
                    penalized_energy_Ha=root['penalized_energy_Ha'],actual_CI_dimension=len(strings)**2,
                    observation_id=oid,evaluation_origin=origin,new_CI_solves=new,CI_numeric_CPU_proxy=proxy,
                    new_actual_CPU_seconds=process_time()-start,new_actual_wall_seconds=perf_counter()-tick)
            except Exception as exc:
                row.update(status='FAILED_PRESERVED',error=repr(exc),traceback=traceback.format_exc(),
                    new_actual_CPU_seconds=process_time()-start,new_actual_wall_seconds=perf_counter()-tick)
            save(RUN/p['cell_id']/'result.json',row);out.append(row)
    finally:db.close()
    if sha256(catalog.read_bytes())!=terminal['catalog_SHA256']:raise RuntimeError('Old catalog mutated')
    save(RUN/'terminal.json',dict(status='ADDITIVE_FREQUENCY_ONLY_TAIL_TERMINAL',plans=len(plans),rows=out,
        CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,new_model_shots=0,new_QPU_shots=0,
        original_T3_states_changed=False,independent_confirmation=False,goal_complete=False))
    print('FREQUENCY_ONLY_TAIL',[(r['case'],r['status']) for r in out],flush=True)


if __name__=='__main__':run()
