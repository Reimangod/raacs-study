"""N40 audit of raw draws, paid selection, saved roots and all final outcomes."""
from io import BytesIO
from pathlib import Path
from time import process_time,perf_counter
import json
import numpy as np
from qiskit import qpy
from .io import load,save,sha256,digest
from .paths import ROOT,OLD
from . import direct_sample_pilot as source
from .finite_source_readout import rank_record,basis_key
from .analyze_finite_source_readout import audit_root


def main():
    cpu,wall=process_time(),perf_counter();p=source.verified();t=load(source.RUN/'terminal.json')
    if t['failure'] or t['actual_model_shots']!=33000 or len(t['draws'])!=33 or len(t['states'])!=39:
        raise RuntimeError('N40 is not fully terminal in its declared pilot scope')
    if t['protocol_SHA256']!=sha256((source.RUN/'protocol.json').read_bytes()):raise ValueError('Protocol binding')
    d=next(x['physical'] for x in p['catalog_all4_definitions'] if x['source']['case']==source.CASE)
    from pyscf.fci import cistring
    sa=cistring.make_strings(range(d['norb']),d['nelec'][0]).astype(np.uint64)
    sb=cistring.make_strings(range(d['norb']),d['nelec'][1]).astype(np.uint64)
    blocked=(sa[:,None]|(sb[None,:]<<np.uint64(d['norb']))).ravel()
    draws={};points={r['point_id']:r for r in p['points']};resources={}
    for row in t['draws']:
        if row['draw_id'] in draws:raise ValueError('Duplicate draw event')
        file=Path(row['path'])
        if sha256(file.read_bytes())!=row['SHA256']:raise ValueError('Raw counts hash')
        q=points[source.point_id(row['point'])]
        intent=load(file.parent/'intent.json')
        if digest(intent)!=row['draw_id'] or any(row[k]!=v for k,v in intent.items()):raise ValueError('Request/result binding')
        expected_seed=source.seed(source.CASE,row['replicate'],row['role'],row['point'] if row['role']=='TRAIN' else None)
        if expected_seed!=row['seed']:raise ValueError('Independent fixed-seed rule')
        with np.load(q['files']['probabilities.npz']['path'],allow_pickle=False) as z:prob=z['probabilities']
        ids,c=np.unique(np.random.default_rng(row['seed']).choice(blocked,size=row['shots'],p=prob),return_counts=True)
        with np.load(file,allow_pickle=False) as z:
            if not np.array_equal(z['blocked_ids'],ids) or not np.array_equal(z['counts'],c) or sum(c)!=row['shots']:
                raise ValueError('Independent raw RNG/count/sector replay differs')
        if rank_record(ids,c,d['norb'],d['nelec'][0])!=row['rank_record']:raise ValueError('Observable-only frequency ranking changed')
        draws[row['draw_id']]=row
    if len({r['seed'] for r in draws.values()})!=33:raise ValueError('Independent seed collision')
    roots={};final=[]
    for r in t['states']:
        draw=draws[r['draw_id']]
        if any(r[k]!=draw[k] for k in ('replicate','role','point')):raise ValueError('CI/source event mismatch')
        if r['arm'] in ('TRAIN_R32','R32_CAP'):R=draw['rank_record']['R32']
        else:R=min(x['rank_record']['R32'] for x in draws.values() if x['replicate']==r['replicate'] and x['role'] in ('FINAL','BASELINE'))
        expected=sorted(draw['rank_record']['rank'][:R])
        if r['R']!=R or r['half_strings']!=expected or r['actual_D']!=R*R:raise ValueError('Actual R/basis rule mismatch')
        pairs=[(a,b) for a in expected for b in expected]
        if basis_key(d,pairs,p['library_SHA256'])!=r['basis_id']:raise ValueError('CI physics/equation identity')
        if r['basis_id'] not in roots:roots[r['basis_id']]=audit_root(r,d,p['library'])
        if r['role']!='TRAIN':final.append(dict(r,**roots[r['basis_id']]))
    if len(roots)!=len(t['attempts']) or len(roots)!=t['actual_new_CI_roots'] or any(x['status']!='COMPLETED' for x in t['attempts']):
        raise ValueError('Actual paid roots/reuse mismatch')
    for rep in source.REPS:
        train=[r for r in t['states'] if r['role']=='TRAIN' and r['replicate']==rep]
        chosen=source.choose(train);selection=next(x for x in t['selections'] if x['replicate']==rep)
        final_draw=next(x for x in t['draws'] if x['replicate']==rep and x['role']=='FINAL')
        if chosen['point']!=selection['point'] or chosen['point']!=final_draw['point'] or chosen['basis_id']!=selection['training_basis_id']:
            raise ValueError('Final outcomes influenced registered training selection')
        if sum(x['shots'] for x in draws.values() if x['replicate']==rep and x['role']!='BASELINE')!=5500:
            raise ValueError('Unpaid training budget')
    for q in p['points']:
        file=q['files']['sampling.qpy'];qc=qpy.load(BytesIO(Path(file['path']).read_bytes()))[0]
        resource={'cnot_count':int(qc.count_ops().get('cx',0)),
                  'cnot_depth':int(qc.depth(filter_function=lambda i:i.operation.name=='cx')),'total_unitary_depth':int(qc.depth())}
        if any(q['resources'][k]!=v for k,v in resource.items()):raise ValueError('Actual output circuit resource mismatch')
        resources[q['point_id']]=resource
    from .analyze import reference
    hf,ref=reference(source.CASE);metadata=load(OLD/'data/evaluation-only'/(source.CASE+'.json'))
    if metadata['problem_id']!=d['problem_id']:raise ValueError('Evaluation-only reference problem mismatch')
    comparisons=[]
    for arm in ('R32_CAP','COMMON_R'):
        rr=[]
        for rep in source.REPS:
            left=next(r for r in final if r['arm']==arm and r['replicate']==rep and r['role']=='FINAL')
            right=next(r for r in final if r['arm']==arm and r['replicate']==rep and r['role']=='BASELINE')
            rr.append({'replicate':rep,'selected_point':left['point'],
                       'selected_energy_Ha':left['energy_Ha'],'baseline_energy_Ha':right['energy_Ha'],
                       'selected_penalized_energy_Ha':left['penalized_energy_Ha'],'baseline_penalized_energy_Ha':right['penalized_energy_Ha'],
                       'delta_energy_mHa':1000*(left['energy_Ha']-right['energy_Ha']),
                       'delta_penalized_energy_mHa':1000*(left['penalized_energy_Ha']-right['penalized_energy_Ha']),
                       'selected_R':left['R'],'baseline_R':right['R'],'selected_D':left['actual_D'],'baseline_D':right['actual_D'],
                       'selected_spin_square':left['spin_square'],'baseline_spin_square':right['spin_square'],
                       'selected_correlation_energy_recovery_pct':100*(hf-left['energy_Ha'])/(hf-ref),
                       'baseline_correlation_energy_recovery_pct':100*(hf-right['energy_Ha'])/(hf-ref),
                       'selected_resources':resources[source.point_id(left['point'])],
                       'baseline_resources':resources[source.point_id((1.,1.))],
                       'each_algorithm_total_model_budget':5500,'selected_final_samples':1000,'baseline_final_samples':5500})
        comparisons.append({'arm':arm,'replicates':rr,'paired_delta_energy_mHa_mean':float(np.mean([r['delta_energy_mHa'] for r in rr])),
                            'paired_delta_energy_mHa_SD':float(np.std([r['delta_energy_mHa'] for r in rr],ddof=1)),
                            'paired_delta_penalized_energy_mHa_mean':float(np.mean([r['delta_penalized_energy_mHa'] for r in rr])),
                            'selected_energy_Ha_mean':float(np.mean([r['selected_energy_Ha'] for r in rr])),
                            'baseline_energy_Ha_mean':float(np.mean([r['baseline_energy_Ha'] for r in rr])),
                            'statistical_superiority_certified':False})
    summary={'status':'N40_ALL33_RAWDRAWS39_ENDPOINTS33_SAVEDROOTS_AND9_QPYS_AUDITED',
             'case':source.CASE,'comparisons':comparisons,'selections':t['selections'],
             'raw_draws_audited':33,'endpoint_records_audited':39,'distinct_roots_audited':len(roots),'actual_new_CI_roots':t['actual_new_CI_roots'],
             'actual_model_samples':t['actual_model_shots'],'actual_QPU_shots':0,'new_FCI':0,'GPU_jobs':0,
             'reference_kind':'FCI_ARCHIVE' if metadata['energies_Ha'].get('fci_energy') is not None else 'SCI_NOT_FCI',
             'reference_precision':'VERIFIED' if metadata.get('sector_and_reference_precision_verified') else 'UNVERIFIED',
             'reference_used_for_selection':False,'configuration_recovery_or_noise_simulated':False,
             'physical_spin_equation':'PROJECTED_PHYSICAL_H_PLUS_0p1S2_NOT_EXACT_S0_NO_OO',
             'other3_cases':'NOT_STARTED_NOT_MISSING_NOT_ALGORITHM_FAILURE',
             'parameter_metric':'TWO_OPTIMIZED_LATENT_SCALES_NOT_TWO_GATE_SLOTS_OR_PROVEN_INDEPENDENT_CIRCUIT_RANK',
             'qualifier_CPU_seconds':load(source.RUN/'qualifier.json')['CPU_seconds'],
             'prepare_CPU_seconds':p['prepare_CPU_seconds'],'pilot_CPU_seconds':t['CPU_seconds'],'pilot_wall_seconds':t['wall_seconds'],
             'old_initialization_cost':next(x['old_initialization_cost'] for x in p['catalog_all4_definitions'] if x['source']['case']==source.CASE),
             'cold_end_to_end_matched_CPU':False,'best_paper_superiority_proven':False,
             'run_bytes':sum(f.stat().st_size for f in source.RUN.rglob('*') if f.is_file()),
             'audit_CPU_seconds':process_time()-cpu,'audit_wall_seconds':perf_counter()-wall}
    save(ROOT/'analysis/N40-direct-sample-summary.json',summary);print(json.dumps(summary))


if __name__=='__main__':main()
