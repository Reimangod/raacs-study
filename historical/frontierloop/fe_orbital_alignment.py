"""N46 read-only archived/fresh CISD RDM gauge check. No eigensolve or sampling."""
from types import SimpleNamespace
from time import process_time, perf_counter
import json
import lzma
import sys
import numpy as np
from pyscf import ao2mo, fci
from pyscf.ci import cisd
from .paths import ROOT, OLD
from .io import load, save, sha256, space

LOCAL = ROOT/'runs/N46-FE-ORBITAL-GAUGE-AUDIT-v1'
ARCHIVE_SHA = '80da3b1d9515c6755f9ef271cfd30368e48d3ac34ff91f3c304edcb1aca99c0e'


def densities(vector, norb, nocc):
    vector = np.asarray(vector, dtype=float)
    if not np.isfinite(vector).all(): raise ValueError('Nonfinite CI vector')
    norm = float(cisd.dot(vector, vector, norb, nocc))
    if abs(norm-1) > 1e-7: raise ValueError('Saved spin-adapted CI norm differs')
    ci = SimpleNamespace(nmo=norb, nocc=nocc, frozen=None,
         mo_coeff=np.eye(norb), mo_occ=np.r_[np.full(nocc, 2.), np.zeros(norb-nocc)],
         stdout=sys.stdout, verbose=0, max_memory=256,
         cisdvec_to_amplitudes=cisd.cisdvec_to_amplitudes)
    d1 = cisd.make_rdm1(ci, vector)
    d2 = cisd.make_rdm2(ci, vector)
    ne = 2*nocc
    errors = [abs(float(np.trace(d1))-ne), abs(float(np.einsum('pprr->', d2))-ne*(ne-1))]
    if max(errors) > 1e-6: raise ValueError('RDM electron traces disagree')
    return d1, d2, dict(spin_adapted_norm=norm, electron_trace_errors=errors)


def expectation(h1, eri, core, d1, d2):
    return float(core + np.einsum('pq,qp->', h1, d1) + .5*np.einsum('pqrs,pqrs->', eri, d2))


def toy():
    rng = np.random.default_rng(4601); n=4; o=2
    c2=rng.normal(size=(o,o,n-o,n-o)); c2=(c2+c2.transpose(1,0,3,2))/2
    v=cisd.amplitudes_to_cisdvec(.3,rng.normal(size=(o,n-o)),c2)
    v/=np.sqrt(cisd.dot(v,v,n,o)); d1,d2,record=densities(v,n,o)
    h=rng.normal(size=(n,n)); h=(h+h.T)/2
    factors=rng.normal(size=(5,n,n)); factors=(factors+factors.transpose(0,2,1))/2
    eri=np.einsum('xpq,xrs->pqrs',factors,factors)
    civec=cisd.to_fcivec(v,n,(o,o))
    exact=float(fci.direct_spin1.energy(h,eri,civec,n,(o,o)))+.17
    rdm=expectation(h,eri,.17,d1,d2)
    error=abs(exact-rdm)
    if error>1e-9: raise ValueError('CISD RDM vs full-vector expectation failed')
    return dict(record, expectation_difference_Ha=error, full_vector_eigensolves=0)


def run():
    if (LOCAL/'terminal.json').exists(): raise RuntimeError('Completed audit exists; inspect rather than rerun')
    space(); cpu,wall=process_time(),perf_counter(); qualification=toy()
    archive=OLD/'data/raw-source'/(ARCHIVE_SHA+'.json.xz')
    if sha256(archive.read_bytes())!=ARCHIVE_SHA: raise ValueError('Pinned archive changed')
    raw=json.loads(lzma.decompress(archive.read_bytes()))
    n=int(raw['norb']); o=int(raw['nelec'][0])
    if (n,o,raw['nelec'][1])!=(20,15,15): raise ValueError('Physical sector differs')
    h=np.asarray(raw['one_body_integrals'],float)
    g=ao2mo.restore(1,np.asarray(raw['two_body_integrals'],float),n)
    c=np.asarray(raw['mo_coeff'],float); core=float(raw['core_energy'])
    ht=c.T@h@c; gt=ao2mo.incore.full(g,c,compact=False).reshape((n,)*4)
    physical=OLD/'data/physical/FE2S2-40-P2'
    with np.load(physical/'integrals.npz',allow_pickle=False) as a:
        errors=[float(np.max(abs(ht-a['h1']))),float(np.max(abs(gt-a['eri'])))]
    if max(errors)>1e-10: raise ValueError('Registered MO Hamiltonian differs')
    fresh=OLD/'data/initialization-fresh/FE2S2-40-P2/amplitudes.npz'
    fresh_meta=load(fresh.with_name('result.json'))
    if sha256(fresh.read_bytes())!=fresh_meta['amplitudes_sha256']: raise ValueError('Fresh CISD input changed')
    fresh_energy=next(x['total_energy_Ha'] for x in fresh_meta['attempts'] if x['method']=='CISD')
    with np.load(fresh,allow_pickle=False) as a: fv=a['cisd_vec'].copy()
    rows=[]
    for name,vec,target in [('ARCHIVED_CISD',raw['cisd_vec'],raw['cisd_energy']),('FRESH_CISD',fv,fresh_energy)]:
        d1,d2,evidence=densities(vec,n,o)
        eraw=expectation(h,g,core,d1,d2); emo=expectation(ht,gt,core,d1,d2)
        rows.append(dict(name=name,recorded_energy_Ha=float(target),raw_gauge_energy_Ha=eraw,
                         registered_MO_energy_Ha=emo,raw_difference_Ha=eraw-target,
                         registered_MO_difference_Ha=emo-target,**evidence))
    if process_time()-cpu>10 or perf_counter()-wall>30: raise RuntimeError('Diagnostic CPU/wall bound exceeded; preserve execution log')
    result=dict(status='N46_ARCHIVED_AND_FRESH_CISD_RDM_GAUGE_AUDITED',rows=rows,
         toy=qualification,archive_SHA256=ARCHIVE_SHA,fresh_amplitudes_SHA256=sha256(fresh.read_bytes()),
         registered_integrals_SHA256=sha256((physical/'integrals.npz').read_bytes()),
         registered_problem_id=load(physical/'problem.json')['problem_id'],
         registered_transform_errors=errors,code_SHA256=sha256(__file__ and open(__file__,'rb').read()),
         new_candidate_energy_evaluations=0, fixed_existing_CISD_expectations=4,
         new_SCF_CCSD_CISD_FCI_eigensolves=0,new_model_samples=0,new_GPU_jobs=0,
         archived_SCI_coefficients_accessed=False,paper_author_error_claim=False,
         source_coefficients_MO_aligned=all(abs(x['registered_MO_difference_Ha'])<1e-7 for x in rows),
         CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,goal_complete=False)
    save(LOCAL/'terminal.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': run()
