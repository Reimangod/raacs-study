"""Exact Z-readout tail extraction with an explicit affine classical decoder.

An engineering baseline, not a claim that this known principle is novel.
Removed gates commute past all later retained gates on disjoint supports.
Their monomial product only permutes outcome labels and adds invisible phases.
"""
from io import BytesIO
from pathlib import Path
from time import process_time
import numpy as np
from qiskit import qpy
from qiskit.circuit.library import (XGate,YGate,CXGate,SwapGate,ZGate,RZGate,PhaseGate,
    SGate,SdgGate,TGate,TdgGate,CZGate,CPhaseGate,RZZGate,IGate)
from .paths import ROOT,OLD
from .io import save,write,sha256


DIAGONAL=(ZGate,RZGate,PhaseGate,SGate,SdgGate,TGate,TdgGate,CZGate,CPhaseGate,RZZGate,IGate)


def extract(circuit):
    if circuit.num_clbits:raise ValueError('Unmeasured unitary sampling circuit required')
    blocked=set();kept=[];removed=[]
    for index in range(len(circuit.data)-1,-1,-1):
        item=circuit.data[index];op=item.operation
        wires=[circuit.find_bit(q).index for q in item.qubits]
        klass=None
        if isinstance(op,DIAGONAL):klass='phase'
        elif isinstance(op,(XGate,YGate)):klass='x'
        elif isinstance(op,CXGate)and op.ctrl_state==1:klass='cx'
        elif isinstance(op,SwapGate):klass='swap'
        if klass is not None and not blocked.intersection(wires):
            removed.append({'source_index':index,'name':op.name,'classical':klass,'wires':wires})
        else:kept.append(item);blocked.update(wires)
    prefix=circuit.copy_empty_like()
    for item in reversed(kept):prefix.append(item.operation,[prefix.qubits[circuit.find_bit(q).index]for q in item.qubits])
    return prefix,list(reversed(removed))


def decode(value,tail,inverse=False):
    x=int(value)
    for gate in reversed(tail)if inverse else tail:
        op=gate['classical'];wires=gate['wires']
        if op=='phase':continue
        if op=='x':x^=1<<wires[0]
        elif op=='cx':
            if x>>wires[0]&1:x^=1<<wires[1]
        elif op=='swap':
            a,b=wires
            if(x>>a&1)!=(x>>b&1):x^=(1<<a)|(1<<b)
        else:raise RuntimeError('Unknown decoder operation')
    return x


def counters(q):
    return {'cnot_count':q.count_ops().get('cx',0),'cnot_depth':q.depth(filter_function=lambda x:x.operation.name=='cx'),
        'total_depth':q.depth()}


def audit():
    results=[];cpu=process_time()
    cases=['NAPH20-BLOCK-BOYS','N2-32-R1p2-P2','N2-32-R2p4-P2','FE2S2-40-P2']
    for case in cases:
        source=OLD/'prepared'/case/'C_LUCJ_COMPRESSED_L1-L1/sampling.qpy'
        q=qpy.load(BytesIO(source.read_bytes()))[0];prefix,tail=extract(q)
        root=ROOT/'readout-tail-v1'/case
        out=BytesIO();qpy.dump(prefix,out);write(root/'prefix.qpy',out.getvalue())
        record={'case_id':case,'method':'SOURCE_BOUND_COMPRESSED_LUCJ_L1_WITH_CLASSICAL_READOUT_TAIL',
            'source_QPY_SHA256':sha256(source.read_bytes()),'compiled_prefix_QPY_SHA256':sha256(out.getvalue()),
            'before':counters(q),'after':counters(prefix),'tail':tail,'tail_classical_operations':sum(g['classical']!='phase'for g in tail),
            'new_model_shots':0,'new_energy_evaluations':0,'kind':'RESOURCE_AND_SYMBOLIC_READOUT_EQUIVALENCE_AUDIT',
            'energy_recovery_preserved_if_prefix_samples_are_decoded':True,
            'prefix_molecular_simulation_or_hardware_experiment_performed':False,
            'source_coefficient_slots_unchanged_conservative':True,'full_state_equivalence_claimed':False,
            'noisy_hardware_superiority_claimed':False,'known_principle_not_new_ansatz_claim':True}
        save(root/'resource-audit.json',record);results.append(record)
        print(case,record['before'],'->',record['after'],'classical',record['tail_classical_operations'],flush=True)
    save(ROOT/'readout-tail-v1/summary.json',{'cases':results,'CPU_seconds':process_time()-cpu,'new_model_shots':0,
        'not_a_full_goal_completion':True})


if __name__=='__main__':audit()
