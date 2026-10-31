"""Integral-driven threshold HCI variational stage (Holmes et al. 2016).

All variational parents participate; no parent/cardinality ranking truncation.
Presorted double-excitation magnitudes implement the paper's actual epsilon
selection. This is not Dice/SHCI, semistochastic PT2 or a ground error bound.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
from time import perf_counter
import numpy as np
from numba import njit
from .integral_hamiltonian import _element
from .study.core import rss_peak


@njit(cache=True)
def double_table(n,h1,eri):
    p=n*(n-1)//2;left=np.empty(p,np.int64);right=np.empty(p,np.int64)
    index=np.full((n,n),-1,np.int64);k=0
    for i in range(n):
        for j in range(i+1,n):left[k]=i;right[k]=j;index[i,j]=k;k+=1
    magnitudes=np.zeros((p,p));computed=0
    for a in range(p):
        i,j=left[a],right[a];ket=(np.uint64(1)<<np.uint64(n-1-i))|(np.uint64(1)<<np.uint64(n-1-j))
        for b in range(p):
            r,s=left[b],right[b]
            if r==i or r==j or s==i or s==j or i%2+j%2!=r%2+s%2:continue
            bra=(np.uint64(1)<<np.uint64(n-1-r))|(np.uint64(1)<<np.uint64(n-1-s))
            magnitudes[a,b]=abs(_element(bra,ket,n,h1,eri,0.));computed+=1
    return left,right,index,magnitudes,computed


@njit(cache=True)
def generate(ket,n,h1,eri,core,left,right,index,magnitudes,order,threshold):
    occ=np.empty(n,np.int64);vir=np.empty(n,np.int64);no=nv=0
    for q in range(n):
        if (ket>>np.uint64(n-1-q))&np.uint64(1):occ[no]=q;no+=1
        else:vir[nv]=q;nv+=1
    cap=no*nv+(no*(no-1)//2)*(nv*(nv-1)//2)
    ids=np.empty(cap,np.uint64);values=np.empty(cap);saved=examined=visits=0
    for ii in range(no):
        i=occ[ii]
        for aa in range(nv):
            a=vir[aa]
            if i%2!=a%2:continue
            bra=ket^(np.uint64(1)<<np.uint64(n-1-i))^(np.uint64(1)<<np.uint64(n-1-a))
            value=_element(bra,ket,n,h1,eri,core);examined+=1
            if abs(value)>threshold:ids[saved]=bra;values[saved]=value;saved+=1
    for ii in range(no):
        i=occ[ii]
        for jj in range(ii+1,no):
            j=occ[jj];pair=index[i,j]
            for rank in range(len(left)):
                b=order[pair,rank];visits+=1
                if magnitudes[pair,b]<=threshold:break
                a,c=left[b],right[b]
                if (ket>>np.uint64(n-1-a))&np.uint64(1) or (ket>>np.uint64(n-1-c))&np.uint64(1):continue
                bra=ket^(np.uint64(1)<<np.uint64(n-1-i))^(np.uint64(1)<<np.uint64(n-1-j))\
                       ^(np.uint64(1)<<np.uint64(n-1-a))^(np.uint64(1)<<np.uint64(n-1-c))
                value=_element(bra,ket,n,h1,eri,core);examined+=1
                if abs(value)>threshold:ids[saved]=bra;values[saved]=value;saved+=1
    return ids[:saved],values[:saved],examined,visits


class HeatBathConnections:
    def __init__(self,backend):
        self.backend=backend;t=perf_counter()
        self.left,self.right,self.index,self.magnitude,computed=double_table(backend.n_qubits,backend.h1,backend.eri)
        self.order=np.argsort(-self.magnitude,axis=1,kind='stable').astype(np.int32)
        self.setup={'wall_seconds':perf_counter()-t,'double_template_entries_evaluated':int(computed),
            'orbital_pair_table_slots':int(self.magnitude.size),
            'table_storage_bytes':sum(x.nbytes for x in (self.left,self.right,self.index,self.magnitude,self.order)),
            'is_Fock_Hamiltonian':False}
        self.examined=self.sorted_visits=0

    def connections(self,ket,threshold):
        if threshold<=0 or not np.isfinite(threshold):raise ValueError('Strictly positive finite cutoff; no FCI limit')
        h=self.backend
        ii,vv,examined,visits=generate(np.uint64(ket),h.n_qubits,h.h1,h.eri,h.core,
            self.left,self.right,self.index,self.magnitude,self.order,float(threshold))
        self.examined+=int(examined);self.sorted_visits+=int(visits);h.matrix_entry_requests+=int(examined)
        return ii,vv


@dataclass(frozen=True)
class HCISpec:
    epsilon1_Ha:float=1e-3
    dimension_cap:int=512
    max_iterations:int=32
    wall_seconds:float=120.
    RSS_cap_bytes:int=2*2**30
    named_entry_cap:int=20000000
    external_id_cap:int=100000


def variational(backend,hf,spec,connections=None,observer=None):
    if spec.epsilon1_Ha<=0 or not 1<=spec.dimension_cap<=512:raise ValueError('Positive threshold and bounded selected space')
    started=perf_counter();connections=connections or HeatBathConnections(backend);selected=[int(hf)];history=[];stop_after_solve=False
    initial_entries=backend.matrix_entry_requests;initial_examined=connections.examined;initial_visits=connections.sorted_visits
    def cap():
        if perf_counter()-started>spec.wall_seconds:return 'WALL_CAP_AT_PARENT_BOUNDARY'
        if rss_peak()>spec.RSS_cap_bytes:return 'RSS_CAP_AT_PARENT_BOUNDARY'
        if backend.matrix_entry_requests-initial_entries>=spec.named_entry_cap:return 'NAMED_ENTRY_CAP_AT_PARENT_BOUNDARY'
    def result(status,partial=None):
        return {'status':status,'spec':asdict(spec),'history':history,'partial_selection':partial,
            'wall_seconds':perf_counter()-started,'peak_process_RSS_hwm':rss_peak(),
            'named_matrix_entry_requests':backend.matrix_entry_requests-initial_entries,
            'connection_entries_examined':connections.examined-initial_examined,
            'sorted_template_visits':connections.sorted_visits-initial_visits,
            'quantum_shots':0,'hardware_shots':0,'FCI_used':False,'PT2_energy_Ha':None,
            'ground_state_error_bound_Ha':None,'logical_CNOT':None,'ground_state_reference':None}
    for iteration in range(spec.max_iterations):
        if why:=cap():return result(why)
        hh=backend.matrix(selected,selected);w,c=np.linalg.eigh(hh);energy=float(w[0]);coeff=c[:,0]
        residual=float(np.linalg.norm(hh@coeff-energy*coeff));assert residual<1e-8
        row={'iteration':iteration,'selected_ids':selected.copy(),'coefficients':coeff.tolist(),
            'dimension':len(selected),'variational_energy_Ha':energy,'internal_Ritz_residual':residual}
        history.append(row)
        if observer:observer(history)
        if stop_after_solve:return result('HCI_VARIATIONAL_SELECTION_LT1_PERCENT')
        if len(selected)==spec.dimension_cap:return result('DIMENSION_CAP')
        old=set(selected);external=set();parents=0
        for determinant,ci in zip(selected,coeff):
            if why:=cap():return result(why,{'complete_parents':parents,'external_ids_found':len(external)})
            if ci!=0:
                ids,values=connections.connections(determinant,spec.epsilon1_Ha/abs(ci))
                external.update(int(x) for x in ids if int(x) not in old)
            parents+=1
            if len(external)>spec.external_id_cap:return result('EXTERNAL_ID_CAP',{'complete_parents':parents,'external_ids_found':len(external)})
        row['all_parent_selection_completed']=True;row['new_threshold_qualified_ids']=len(external)
        if not external:return result('HCI_VARIATIONAL_SELECTION_NO_NEW_IDS')
        if len(selected)+len(external)>spec.dimension_cap:
            return result('DIMENSION_CAP_BEFORE_COMPLETE_THRESHOLD_EXPANSION',{'complete_parents':parents,'external_ids_found':len(external)})
        stop_after_solve=len(external)<.01*len(selected)
        selected=sorted(old|external)
    return result('ITERATION_CAP')
