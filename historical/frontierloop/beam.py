"""Residual-guided bounded path exploration, without immediate-gain stopping.

Ritz vectors guide proposals only. Every quantum-model observation comes from
an actual author-QE circuit; no classical CI state is prepared for free.
"""
from dataclasses import asdict
from time import process_time
import numpy as np
from benchmark.local_sampler_primitives import local_proposals
from benchmark.ansatz import resources,measurement_equivalent_circuit
from paperstudy.ci import interleaved_to_blocked
from raacsnext.policy import BoundaryStop,numeric_choice,resource_key,action_key
from raacsnext.sampler import merge_counts
from .bank import BankEngine
from .io import digest,save
from .journal import circuit_id


def activation_fraction(op,observed):
    """Occupation-pattern activation, not an estimate of unknown phases."""
    n=op.n_qubits
    src=sum(1<<(n-1-q)for q in op.source);dst=sum(1<<(n-1-q)for q in op.target)
    active=sum(count for x,count in observed.items()if((x&src)==src and not(x&dst))or((x&dst)==dst and not(x&src)))
    return active/sum(observed.values())


def spin_strings(member,norb):
    mask=(1<<norb)-1;strings=set()
    for raw in member['observed']:
        x=interleaved_to_blocked(raw,norb);strings.update((x&mask,x>>norb))
    return strings


class BeamEngine(BankEngine):
    def __init__(self,meta,h,g,journal,directory,budget,CPU_cap):
        super().__init__(meta,h,g,journal,directory,budget,CPU_cap,64,64)
        self.archive={};self.proposal_calls=0

    def record_measure(self,actions,shots,role):
        member=self.measure(actions,shots,role)
        self.archive=merge_counts([self.archive,member['observed']])
        return member

    def global_proposals(self,parents):
        # A classical vector fitted only to paid observed configurations.
        archive_member={'observed':self.archive}
        ci=self.raw_ci(archive_member)
        proposed,scope=local_proposals(self.backend,self.connections,ci,128,32,False)
        self.proposal_calls+=1
        return proposed,scope

    def parent_proposals(self,parent,proposed):
        ranked=[]
        for p in proposed:
            self.boundary();activation=activation_fraction(p['operator'],parent['observed'])
            if activation<=0:continue
            q=p['operator'].append(parent['circuit'],np.pi/8);q,_=measurement_equivalent_circuit(q)
            rr=resources(q,len(parent['actions'])+1,len(parent['actions'])+1)
            old=parent['metadata']['sampling_resources']
            cost=1+max(0,rr['cnot_count']-old['cnot_count'])+.5*max(0,rr['cnot_depth']-old['cnot_depth'])
            ranked.append((p['residual_priority']*activation/cost,p))
        if not ranked:
            # Same bounded fallback for every Hamiltonian, never a molecule table.
            return self.proposals(parent)[0][:2]
        return[p for _,p in sorted(ranked,key=lambda z:(-z[0],z[1]['operator']))[:2]]

    def select_bank(self,candidates):
        remaining=list(candidates);selected=[]
        for _ in range(min(4,len(remaining))):
            self.boundary()
            winner=numeric_choice(remaining,lambda m:self.energy(selected+[m],64),
                lambda m:(resource_key(selected+[m]),tuple(x['id']for x in selected+[m])))
            selected.append(winner);remaining=[m for m in remaining if m['id']!=winner['id']]
        return sorted(selected,key=action_key)

    def run(self):
        hf=self.record_measure([],32,'initial-HF')
        frontier=[hf];best_bank=[hf];best_energy=self.energy(best_bank,64)
        stop='DEPTH_CAP';seen={hf['id']};completed_depth=0
        for depth in range(1,9):
            record={'depth':depth,'frontier_ids':[m['id']for m in frontier],'completed':False}
            try:
                self.boundary();proposals,scope=self.global_proposals(frontier);candidates=[]
                for parent in sorted(frontier,key=action_key):
                    for p in self.parent_proposals(parent,proposals):
                        for angle in(-np.pi/4,-np.pi/8,np.pi/8,np.pi/4):
                            actions=parent['actions']+[{**asdict(p['operator']),'theta':float(angle)}]
                            cid=circuit_id(self.meta,actions)
                            if cid in seen:continue
                            child=self.record_measure(actions,64,['beam-screen',depth,parent['id'],p['operator'].identity,float(angle)])
                            seen.add(cid)
                            score=self.energy([child],16)
                            candidates.append({'member':child,'score':score})
                if not candidates:stop='NO_ACTIVE_NEW_PROPOSALS';break
                # Bounded coarse-to-fine selection. All measurements, including
                # candidates not retained here, still enter the paid archive.
                old_strings=set().union(*(spin_strings(m,self.meta['norb'])for m in frontier))
                pool=list(candidates);chosen=[]
                for _ in range(min(8,len(pool))):
                    winner=numeric_choice(pool,lambda x:x['score'],lambda x:(
                        -len(spin_strings(x['member'],self.meta['norb'])-old_strings),
                        resource_key([x['member']]),x['member']['id']))
                    chosen.append(winner['member']);pool=[x for x in pool if x['member']['id']!=winner['member']['id']]
                frontier=self.select_bank(chosen)
                value=self.energy(frontier,64)
                # Search continues even if this stage is temporarily worse.
                if(value<best_energy-1e-8 or(abs(value-best_energy)<=1e-8 and
                    (resource_key(frontier),tuple(m['id']for m in frontier))<
                    (resource_key(best_bank),tuple(m['id']for m in best_bank)))):
                    best_bank=list(frontier);best_energy=value
                completed_depth=depth
                record.update(completed=True,candidates_measured=len(candidates),preselected=len(chosen),
                    proposal_scope=scope,frontier_energy_Ha=value,best_pilot_energy_Ha=best_energy,
                    best_bank_ids=[m['id']for m in best_bank],pilot_used=self.pilot_used,
                    construction_CPU_seconds=process_time()-self.cpu)
                self.decisions.append(record);save(self.directory/'stages'/(str(depth)+'.json'),record)
            except BoundaryStop as exc:
                stop=str(exc);record.update(boundary_stop=stop,pilot_used=self.pilot_used)
                self.decisions.append(record);save(self.directory/'stages'/(str(depth)+'.json'),record);break
        construction_cpu=process_time()-self.cpu;remaining=self.budget-self.pilot_used
        if remaining<self.budget/2:raise RuntimeError('Independent final reserve violated')
        final=[];best_bank=sorted(best_bank,key=action_key)
        for i,m in enumerate(best_bank):
            shots=remaining//len(best_bank)+(i<remaining%len(best_bank))
            child=self.journal.measure(self.meta,m['actions'],shots,['independent-final',i,m['id']])
            final.append({'member':child,'shots':shots})
        return {'final_members':final,'pilot_model_shots':self.pilot_used,'construction_stop_reason':stop,
            'construction_CPU_seconds':construction_cpu,'construction_CPU_cap':self.CPU_cap,
            'completed_depth':completed_depth,'decision_count':len(self.decisions),'bank_size':len(best_bank),
            'best_pilot_bank_energy_Ha_selection_biased':best_energy,
            'raw_observed_archive_unique':len(self.archive),'proposal_calls':self.proposal_calls,
            'ci_calls':self.ci_calls,'ci_matvecs':self.ci_matvecs,'ci_cache_hits':self.ci_cache_hits,
            'named_H_requests':self.backend.matrix_entry_requests,'reference_information_used':False,
            'molecule_label_used':False,'temporary_non_improving_paths_allowed':True}
