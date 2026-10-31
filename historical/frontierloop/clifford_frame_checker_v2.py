"""Gate-level sparse simulation in a moving Clifford frame.

Cliffords are tracked by a tableau; actual Rx/Ry/Rz gates become conjugated
Pauli rotations. Source Pauli sequences and sector projections are NOT inputs.
The final Clifford is applied sparsely under the SAME bounded support cap.
This is a bounded numerical probability verifier, not an exact formal proof.
"""
import base64,gzip,json,math,sys
from io import BytesIO
from time import process_time,perf_counter
import numpy as np
from qiskit import QuantumCircuit,qpy
from qiskit.quantum_info import Clifford,Pauli


def masks(p):
    if int(p.phase) not in (0,2):raise ValueError('Non-Hermitian conjugated rotation')
    x=sum(int(b)<<i for i,b in enumerate(p.x));z=sum(int(b)<<i for i,b in enumerate(p.z))
    return x,z,(-1 if int(p.phase)==2 else 1)*(1j**((x&z).bit_count()))


def probabilities(qc,CPU_seconds=20,support_cap=200000,amplitude_cutoff=1e-14):
    if qc.num_qubits>40 or qc.num_clbits or len(qc.data)>100000:raise ValueError('Bounded circuit scope')
    cpu,wall=process_time(),perf_counter();n=qc.num_qubits;frame=Clifford(QuantumCircuit(n));state={0:1.+0j}
    discarded=0.;peak=1;rotations=0;cliffords=0
    for inst in qc.data:
        if process_time()-cpu>CPU_seconds:raise TimeoutError('Finite frame simulation CPU cap')
        op=inst.operation;qs=[qc.find_bit(q).index for q in inst.qubits]
        if op.name in ['barrier','id']:continue
        if op.name in ['rx','ry','rz']:
            if len(qs)!=1:raise ValueError('Rotation arity')
            label=['I']*n;label[n-1-qs[0]]=op.name[1].upper()
            p=Pauli(''.join(label)).evolve(frame,frame='h');x,z,phase=masks(p)
            a=float(op.params[0])/2
            if not math.isfinite(a):raise ValueError('Nonfinite angle')
            c=math.cos(a);s=-1j*math.sin(a)*phase
            new={k:c*v for k,v in state.items()}
            for k,v in state.items():
                dest=k^x;new[dest]=new.get(dest,0j)+s*(-1 if (z&k).bit_count()%2 else 1)*v
            dropped=[abs(v)**2 for v in new.values() if abs(v)<amplitude_cutoff]
            discarded+=math.sqrt(math.fsum(dropped))
            state={k:v for k,v in new.items() if abs(v)>=amplitude_cutoff}
            if len(state)>support_cap:raise MemoryError('Sparse support cap; no larger fallback')
            if discarded>1e-10:raise ValueError('Accumulated discarded amplitude bound exceeded')
            rotations+=1;peak=max(peak,len(new))
        elif op.name in ['h','s','sdg','x','y','z','cx','cz','swap','sx','sxdg']:
            step=QuantumCircuit(len(qs));step.append(op,range(len(qs)))
            frame=frame.compose(Clifford(step),qargs=qs);cliffords+=1
        else:raise ValueError('Unsupported primitive '+op.name)
    # For a monomial final Clifford, each input address maps injectively to one
    # output address. Its phase cannot affect these outcome probabilities.
    image=[]
    nonmonomial=False
    for q in range(n):
        label=['I']*n;label[n-1-q]='Z'
        x,z,phase=masks(Pauli(''.join(label)).evolve(frame,frame='h'))
        if x or phase not in (1,-1):nonmonomial=True
        image.append((z,phase==-1))
    if nonmonomial:
        # Bounded computational-basis application of the final Clifford.
        # This may be inefficient; never substitute a fabricated distribution.
        final_circuit=frame.to_circuit()
        for inst in final_circuit.data:
            if process_time()-cpu>CPU_seconds:raise TimeoutError('Final Clifford CPU cap')
            name=inst.operation.name
            # Qubit indices in synthesized Clifford are its register order.
            qs=[final_circuit.find_bit(q).index for q in inst.qubits]
            new={}
            for k,v in state.items():
                dest=k;factor=1.+0j
                if name=='h':
                    q=qs[0];bit=(k>>q)&1
                    new[k]=new.get(k,0j)+v*((-1)**bit)/math.sqrt(2)
                    dest=k^(1<<q);factor=1/math.sqrt(2)
                elif name in ['s','sdg','z']:
                    if (k>>qs[0])&1:factor={'s':1j,'sdg':-1j,'z':-1}[name]
                elif name in ['x','y']:
                    dest=k^(1<<qs[0])
                    if name=='y':factor=-1j if ((k>>qs[0])&1) else 1j
                elif name=='cx':
                    if (k>>qs[0])&1:dest=k^(1<<qs[1])
                elif name=='cz':
                    if ((k>>qs[0])&1) and ((k>>qs[1])&1):factor=-1
                elif name=='swap':
                    if ((k>>qs[0])^(k>>qs[1]))&1:dest=k^(1<<qs[0])^(1<<qs[1])
                elif name in ['id','barrier']:pass
                else:raise ValueError('Unsupported synthesized Clifford '+name)
                new[dest]=new.get(dest,0j)+factor*v
            discarded+=math.sqrt(math.fsum(abs(v)**2 for v in new.values() if abs(v)<amplitude_cutoff))
            state={k:v for k,v in new.items() if abs(v)>=amplitude_cutoff}
            peak=max(peak,len(new))
            if len(state)>support_cap:raise MemoryError('Final Clifford support cap')
            if discarded>1e-10:raise ValueError('Final discarded norm cap')
        output={k:float(abs(v)**2) for k,v in state.items()}
    else:
        output={}
        for k,v in state.items():
            address=sum((((z&k).bit_count()%2)^int(sign))<<q for q,(z,sign) in enumerate(image))
            if address in output:raise ValueError('Final Clifford address collision')
            output[address]=float(abs(v)**2)
    return output,dict(CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
        final_Clifford_nonmonomial=nonmonomial,source_qubits=n,primitive_gates=len(qc.data),rotation_gates=rotations,Clifford_gates=cliffords,
        final_support=len(state),peak_sparse_support=peak,amplitude_cutoff=amplitude_cutoff,
        discarded_amplitude_L2_sum=discarded,discarding_only_probability_L1_bound=2*discarded+discarded**2,
        floating_point_roundoff_not_formally_bounded=True,sector_projection_used=False,
        source_Pauli_sequence_used=False,full_dense_statevector_used=False)


def check(request):
    if request['tolerance']!=1e-8:raise ValueError('Unchanged acceptance tolerance required')
    cpu,wall=process_time(),perf_counter();rows={}
    for name,item in request['circuits'].items():
        q=qpy.load(BytesIO(gzip.decompress(base64.b64decode(item['qpy_gz_base64']))))[0]
        got,info=probabilities(q,CPU_seconds=max(.001,25-(process_time()-cpu)))
        expected={int(k):float(v) for k,v in item['expected_probabilities'].items()}
        if not expected or any(not math.isfinite(v) or v<0 for v in expected.values()) or abs(sum(expected.values())-1)>1e-8:
            raise ValueError('Invalid reference distribution')
        norm=math.fsum(got.values());outside=math.fsum(v for k,v in got.items() if k not in expected)
        l1=math.fsum(abs(got.get(k,0)-v) for k,v in expected.items())+outside
        bound=info['discarding_only_probability_L1_bound']
        if not math.isfinite(l1) or l1+bound>1e-8 or abs(norm-1)+bound>1e-8:
            raise ValueError('Frame/source probability mismatch '+str((l1,bound,norm)))
        rows[name]=dict(info,status='GATE_LEVEL_CLIFFORD_FRAME_PROBABILITY_PASSED',
            probability_L1_with_leakage_bound=l1,probability_L1_plus_discard_bound=l1+bound,
            norm_squared=norm,outside_support_mass=outside)
    return dict(circuits=rows,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
        new_model_samples=0,new_QPU_shots=0,noise_or_unitary_equivalence=False)


if __name__=='__main__':
    import resource
    resource.setrlimit(resource.RLIMIT_CPU,(30,30))
    print(json.dumps(check(json.load(sys.stdin)),allow_nan=False))
