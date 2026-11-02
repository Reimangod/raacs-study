"""D1 stop confirmation, D2 aligned CI, D3 bounded complementary circuit bank.

Input consists only of physical integrals/HF, paid pilot observations, and caps.
No molecular labels, evaluation references, hidden amplitudes or learned scores.
"""
from dataclasses import asdict
from time import process_time, perf_counter

import numpy as np
from benchmark.integral_connections import IntegralConnections
from benchmark.integral_hamiltonian import IntegralHamiltonian
from benchmark.local_sampler_primitives import local_proposals, observed_ci
from benchmark.ansatz import resources, measurement_equivalent_circuit
from paperstudy.ci import solve, interleaved_to_blocked
from .sampler import balanced_counts
from .io import digest, save


class BoundaryStop(RuntimeError):
    pass


def action_key(member):
    return member['id']


def resource_key(bank):
    values = [m['metadata']['sampling_resources'] for m in bank]
    return tuple(float(np.mean([v[k] for v in values])) for k in
        ('cnot_depth','cnot_count','total_depth')) + (float(np.mean([len(m['actions']) for m in bank])),)


def normalized_resources(member):
    # Old ansatz helper calls this total_depth; final LUCJ views call it
    # total_unitary_depth. Keep one explicit internal representation.
    for kind in ('sampling_resources','preparation_resources'):
        value = member['metadata'][kind]
        if 'total_depth' not in value and 'total_unitary_depth' in value:
            value['total_depth'] = value['total_unitary_depth']
    return member


def numeric_choice(items, energy, tail):
    best = min(energy(x) for x in items)
    return min((x for x in items if energy(x) <= best+1e-8),key=tail)


class Engine:
    def __init__(self, meta, h1, eri, journal, directory, variant, budget, CPU_cap):
        if variant not in ('D1','D2','D3'):
            raise ValueError('Explicit registered successor variant required')
        self.meta = {k:meta[k] for k in ('problem_id','norb','n_qubits','hf_id','hf_energy_Ha','core_energy','nalpha','nbeta')}
        self.h1,self.eri = h1,eri
        self.backend = IntegralHamiltonian.from_saved_integrals(h1,eri,meta['hf_id'],meta['hf_energy_Ha'])
        self.connections = IntegralConnections(self.backend,parent_cap=128)
        self.journal,self.directory = journal,directory
        self.variant,self.budget,self.CPU_cap = variant,int(budget),float(CPU_cap)
        self.cpu,self.wall = process_time(),perf_counter()
        self.construction_cap = self.budget-int(np.ceil(self.budget*.5))
        self.pilot_used = 0
        self.cache = {}
        self.ci_calls=self.ci_cache_hits=self.ci_matvecs=0
        self.decisions=[]
        self.seen=set()

    def boundary(self, shots=0):
        if self.pilot_used+shots > self.construction_cap:
            raise BoundaryStop('PILOT_BUDGET_CAP')
        if process_time()-self.cpu >= self.CPU_cap:
            raise BoundaryStop('CONSTRUCTION_CPU_CAP')
        if self.backend.matrix_entry_requests >= 20000000:
            raise BoundaryStop('NAMED_H_WORK_CAP')
        if perf_counter()-self.wall > 1200:
            raise BoundaryStop('CONSTRUCTION_WALL_CAP')

    def measure(self, actions, shots, role):
        self.boundary(shots)
        member = normalized_resources(self.journal.measure(self.meta,actions,shots,role))
        self.pilot_used += shots
        return member

    def raw_ci(self, member):
        key=('raw',tuple(sorted(member['observed'].items())))
        if key not in self.cache:
            self.boundary()
            m=self.meta
            self.cache[key]=observed_ci(self.backend,member['observed'],m['hf_id'],m['nalpha'],m['nbeta'],64)
            self.ci_calls+=1
        else:self.ci_cache_hits+=1
        return self.cache[key]

    def energy(self, bank, r=8):
        if self.variant=='D1':
            if len(bank)!=1:raise ValueError('D1 is single-circuit')
            return self.raw_ci(bank[0])['energy_Ha']
        self.boundary()
        order=sorted(bank,key=action_key)
        counts,subset=balanced_counts([x['observed'] for x in order],
            int(digest(['balanced-pilot',self.journal.run_id,[(m['id'],sorted(m['observed'].items()))for m in order]])[:16],16))
        key=('Cartesian',r,tuple(sorted(counts.items())))
        if key not in self.cache:
            ids=np.array([interleaved_to_blocked(x,self.meta['norb'])for x in sorted(counts)],np.uint64)
            values=np.array([counts[x]for x in sorted(counts)],np.uint64)
            c=solve(self.h1,self.eri,self.meta['core_energy'],(self.meta['nalpha'],self.meta['nbeta']),ids,values,r,seconds=30)
            self.cache[key]=c;self.ci_calls+=1;self.ci_matvecs+=c['matvec_calls']
        else:self.ci_cache_hits+=1
        return self.cache[key]['energy_Ha']

    def combine(self, bank, child, r):
        if self.variant!='D3':return [child]
        combined={m['id']:m for m in bank}
        combined[child['id']]=child
        combined=sorted(combined.values(),key=action_key)
        choices=([m for m in combined if m['id']!=drop['id']]for drop in combined)if len(combined)>4 else [combined]
        return numeric_choice(list(choices),lambda b:self.energy(b,r),
            lambda b:(resource_key(b),tuple(m['id']for m in b)))

    def proposals(self, parent):
        self.boundary()
        ci=self.raw_ci(parent)
        candidates,scope=local_proposals(self.backend,self.connections,ci,128,32,False)
        parent_resources=parent['metadata']['sampling_resources']
        amplitudes=dict(zip(ci['selected_ids'],ci['coefficients']))
        for p in candidates:
            self.boundary()
            q=p['operator'].append(parent['circuit'],np.pi/8)
            q,_=measurement_equivalent_circuit(q)
            value=resources(q,len(parent['actions'])+1,len(parent['actions'])+1)
            cost=1+max(0,value['cnot_count']-parent_resources['cnot_count'])+.5*max(0,value['cnot_depth']-parent_resources['cnot_depth'])
            action_ids=set(p['operator'].action(amplitudes))
            novelty=len(action_ids-set(parent['observed'])-{self.meta['hf_id']})/max(1,len(action_ids))
            p['priority']=p['residual_priority']*(1+novelty)/cost
        return sorted(candidates,key=lambda p:(-p['priority'],p['operator']))[:4],scope

    def choose(self, measured, base_energy):
        eligible=[m for m in measured if base_energy-m['energy']>1e-6]
        if not eligible:return None
        if self.variant=='D1':
            return max(eligible,key=lambda m:(m['utility'],base_energy-m['energy'],
                -m['child']['metadata']['sampling_resources']['cnot_count'],
                tuple(m['child']['actions'][-1]['source']),tuple(m['child']['actions'][-1]['target']),m['child']['actions'][-1]['theta']))
        return numeric_choice(eligible,lambda m:m['energy'],lambda m:(resource_key(m['bank']),m['child']['id']))

    def run(self):
        bank=[self.measure([],32,'initial-HF')]
        self.seen.add(bank[0]['id'])
        stop='DECISION_CAP'
        accepted=0
        cursor=0
        for step in range(1,33):
            record={'step':step,'accepted':False,'pilot_before':self.pilot_used,'variant':self.variant}
            try:
                self.boundary()
                available=[m for m in sorted(bank,key=action_key)if self.variant!='D3' or len(m['actions'])<8]
                if not available:stop='MEMBER_ACTION_CAP';break
                parent=available[cursor%len(available)];cursor+=1
                proposals,scope=self.proposals(parent)
                if not proposals:stop='NO_LOCAL_PROPOSALS';break
                base=self.energy(bank,8)
                measured=[]
                for p in proposals:
                    for angle in(-np.pi/4,-np.pi/8,np.pi/8,np.pi/4):
                        actions=parent['actions']+[{**asdict(p['operator']),'theta':float(angle)}]
                        cid=digest(['physical-QE-actions',self.meta['hf_id'],self.meta['n_qubits'],actions])
                        # D1 preserves original candidate repetitions. New bank
                        # policy never spends twice on an identical action word.
                        if self.variant=='D3' and cid in self.seen:continue
                        before=self.backend.matrix_entry_requests
                        child=self.measure(actions,32,['screen',step,p['operator'].identity,float(angle)])
                        self.seen.add(cid)
                        candidate_bank=self.combine(bank,child,8)
                        energy=self.energy(candidate_bank,8)
                        dc=max(0,child['metadata']['sampling_resources']['cnot_count']-parent['metadata']['sampling_resources']['cnot_count'])
                        gain=base-energy
                        utility=gain/(1+dc+(self.backend.matrix_entry_requests-before)/64)
                        measured.append({'child':child,'bank':candidate_bank,'energy':energy,'utility':utility})
                winner=self.choose(measured,base)
                record.update(proposal_scope=scope,screen_pairs=len(measured),base_pilot_energy_Ha=base,
                    measured=[{'circuit_id':m['child']['id'],'pilot_energy_Ha':m['energy'],'utility':m['utility']}for m in measured],
                    confirmation_requested=winner is None and bool(measured),confirmation_used=False)
                if winner is None and measured:
                    # One fresh confirmation pass, never repeated until lucky.
                    if self.variant=='D1':
                        ranked=sorted(measured,key=lambda m:(-m['utility'],-(base-m['energy']),m['child']['id']))[:2]
                    else:
                        ranked=sorted(measured,key=lambda m:(m['energy'],resource_key(m['bank']),m['child']['id']))[:2]
                    required=128*(len(bank)+len(ranked))
                    self.boundary(required)
                    fresh_bank=[self.measure(m['actions'],128,['confirm-parent',step,m['id']])for m in sorted(bank,key=action_key)]
                    record['confirmation_used']=True
                    cr=16 if self.variant in('D2','D3')else 8
                    confirmed_base=self.energy(fresh_bank,cr)
                    confirmed=[]
                    for m in ranked:
                        before=self.backend.matrix_entry_requests
                        child=self.measure(m['child']['actions'],128,['confirm-child',step,m['child']['id']])
                        b=self.combine(fresh_bank,child,cr)
                        e=self.energy(b,cr)
                        dc=max(0,child['metadata']['sampling_resources']['cnot_count']-parent['metadata']['sampling_resources']['cnot_count'])
                        confirmed.append({'child':child,'bank':b,'energy':e,'utility':(confirmed_base-e)/
                            (1+dc+(self.backend.matrix_entry_requests-before)/64)})
                    winner=self.choose(confirmed,confirmed_base)
                    record.update(confirmation_r=cr,confirmation_base_Ha=confirmed_base,
                        confirmation_children=[{'circuit_id':m['child']['id'],'energy_Ha':m['energy']}for m in confirmed])
                if winner is None:
                    stop='NO_CONFIRMED_PILOT_IMPROVEMENT'
                    self.decisions.append(record);save(self.directory/'decisions'/(str(step)+'.json'),record)
                    break
                bank=winner['bank'];accepted+=1
                record.update(accepted=True,bank_ids=[m['id']for m in bank],winner_id=winner['child']['id'],
                    pilot_used=self.pilot_used,construction_CPU_seconds=process_time()-self.cpu)
                self.decisions.append(record);save(self.directory/'decisions'/(str(step)+'.json'),record)
            except BoundaryStop as exc:
                stop=str(exc);record.update(boundary_stop=stop,pilot_used=self.pilot_used)
                self.decisions.append(record);save(self.directory/'decisions'/(str(step)+'.json'),record)
                break
        construction_cpu=process_time()-self.cpu
        remaining=self.budget-self.pilot_used
        if remaining < int(np.ceil(.5*self.budget)):
            raise RuntimeError('Independent final reserve violated')
        bank=sorted(bank,key=action_key)
        assignments=[remaining//len(bank)+(i<remaining%len(bank))for i in range(len(bank))]
        final=[]
        for i,(m,shots)in enumerate(zip(bank,assignments)):
            if not shots:continue
            child=normalized_resources(self.journal.measure(self.meta,m['actions'],shots,['independent-final',i,m['id']]))
            final.append({'member':child,'shots':int(shots)})
        return {'final_members':final,'pilot_model_shots':self.pilot_used,
            'construction_stop_reason':stop,'accepted_decisions':accepted,
            'construction_CPU_seconds':construction_cpu,'construction_CPU_cap':self.CPU_cap,
            'named_H_requests':self.backend.matrix_entry_requests,'ci_calls':self.ci_calls,
            'ci_cache_hits':self.ci_cache_hits,'ci_matvecs':self.ci_matvecs,
            'decision_count':len(self.decisions),'bank_size':len(bank),
            'reference_information_used':False,'molecule_label_used':False}
