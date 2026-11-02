"""One-variable successor: complete parent passes, old D3 numerical rules."""
from collections import deque
from dataclasses import asdict
from time import process_time
import numpy as np
from raacsnext.policy import Engine as PreviousEngine,BoundaryStop,action_key,resource_key
from .io import digest,save
from .journal import circuit_id


class ParentPass:
    """Stable parent identity; no mutable-list-index round-robin."""
    def __init__(self):self.context=None;self.queue=deque();self.tried=set()
    def next(self,bank):
        context=tuple(sorted(m['id']for m in bank))
        if context!=self.context:
            self.context=context;self.queue=deque(context);self.tried=set()
        members={m['id']:m for m in bank}
        while self.queue:
            key=self.queue.popleft();self.tried.add(key)
            if key in members and len(members[key]['actions'])<8:return members[key]
        return None


class BankEngine(PreviousEngine):
    def __init__(self,meta,h,g,journal,directory,budget,CPU_cap,screen_r=8,confirm_r=16):
        super().__init__(meta,h,g,journal,directory,'D3',budget,CPU_cap)
        self.screen_r,self.confirm_r=screen_r,confirm_r
        self.saved_candidates={};self.reused_candidates=0

    def screen(self,actions,step,op,angle):
        key=circuit_id(self.meta,actions)
        if key in self.saved_candidates:
            self.reused_candidates+=1
            return self.saved_candidates[key]
        child=self.measure(actions,32,['screen',step,op.identity,float(angle)])
        self.saved_candidates[key]=child
        return child

    def run(self):
        bank=[self.measure([],32,'initial-HF')];order=ParentPass();accepted=0;stop='DECISION_CAP'
        for step in range(1,33):
            record={'step':step,'accepted':False,'pilot_before':self.pilot_used}
            try:
                self.boundary();parent=order.next(bank)
                if parent is None:stop='ALL_CURRENT_PARENTS_STAGNANT_OR_MAX_LENGTH';break
                record.update(parent_id=parent['id'],parent_length=len(parent['actions']),bank_before=[m['id']for m in bank])
                proposals,scope=self.proposals(parent)
                base=self.energy(bank,self.screen_r);measured=[]
                for p in proposals:
                    for angle in(-np.pi/4,-np.pi/8,np.pi/8,np.pi/4):
                        actions=parent['actions']+[{**asdict(p['operator']),'theta':float(angle)}]
                        child=self.screen(actions,step,p['operator'],angle)
                        candidate_bank=self.combine(bank,child,self.screen_r)
                        energy=self.energy(candidate_bank,self.screen_r)
                        measured.append({'child':child,'bank':candidate_bank,'energy':energy,'utility':base-energy})
                winner=self.choose(measured,base)
                record.update(base_energy_Ha=base,screen_pairs=len(measured),proposal_scope=scope,
                    confirmation_used=False,reused_candidates_total=self.reused_candidates,
                    measured=[{'child_id':m['child']['id'],'energy_Ha':m['energy']}for m in measured])
                if winner is None and measured:
                    ranked=sorted(measured,key=lambda m:(m['energy'],resource_key(m['bank']),m['child']['id']))[:2]
                    self.boundary(128*(len(bank)+len(ranked)))
                    fresh=[self.measure(m['actions'],128,['confirm-parent',step,m['id']])for m in sorted(bank,key=action_key)]
                    baseline=self.energy(fresh,self.confirm_r);confirmed=[]
                    for m in ranked:
                        child=self.measure(m['child']['actions'],128,['confirm-child',step,m['child']['id']])
                        b=self.combine(fresh,child,self.confirm_r);energy=self.energy(b,self.confirm_r)
                        confirmed.append({'child':child,'bank':b,'energy':energy,'utility':baseline-energy})
                    winner=self.choose(confirmed,baseline)
                    record.update(confirmation_used=True,confirmation_base_Ha=baseline,
                        confirmation_children=[{'child_id':m['child']['id'],'energy_Ha':m['energy']}for m in confirmed])
                if winner is None:
                    record['parent_stagnant']=True
                    # Do not stop globally: another current parent can expand.
                else:
                    bank=winner['bank'];accepted+=1
                    record.update(accepted=True,winner_id=winner['child']['id'],bank_after=[m['id']for m in bank])
                record.update(pilot_used=self.pilot_used,construction_CPU_seconds=process_time()-self.cpu)
                self.decisions.append(record);save(self.directory/'decisions'/(str(step)+'.json'),record)
            except BoundaryStop as exc:
                stop=str(exc);record.update(boundary_stop=stop,pilot_used=self.pilot_used)
                self.decisions.append(record);save(self.directory/'decisions'/(str(step)+'.json'),record);break
        construction_cpu=process_time()-self.cpu
        remaining=self.budget-self.pilot_used
        if remaining<self.budget/2:raise RuntimeError('Independent final reserve violated')
        bank=sorted(bank,key=action_key);final=[]
        for i,m in enumerate(bank):
            shots=remaining//len(bank)+(i<remaining%len(bank))
            member=self.journal.measure(self.meta,m['actions'],shots,['independent-final',i,m['id']])
            final.append({'member':member,'shots':shots})
        return {'final_members':final,'pilot_model_shots':self.pilot_used,'construction_stop_reason':stop,
            'construction_CPU_seconds':construction_cpu,'construction_CPU_cap':self.CPU_cap,
            'accepted_decisions':accepted,'bank_size':len(bank),'decision_count':len(self.decisions),
            'reused_candidate_observations':self.reused_candidates,'ci_calls':self.ci_calls,
            'ci_matvecs':self.ci_matvecs,'ci_cache_hits':self.ci_cache_hits,
            'named_H_requests':self.backend.matrix_entry_requests,'reference_information_used':False,
            'molecule_label_used':False,'screen_r':self.screen_r,'confirmation_r':self.confirm_r}
