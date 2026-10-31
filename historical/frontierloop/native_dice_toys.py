"""Tiny synthetic Hamiltonians to qualify native basis/core/spin/HCI semantics.

The output binary convention follows the pinned Qiskit addon (Apache-2.0).
The pinned DICE writer ALREADY applies Dice-to-alpha/beta fermionic parity;
applying that sign a second time would be wrong. No production references or
scientific molecular candidates are used in this qualification.
"""
import argparse,itertools,json,math,struct
from pathlib import Path
import numpy as np
from .paths import ROOT
from .io import load,save,write,sha256

RUN=ROOT/'runs/N2-NATIVE-DICE-TOYS-v1'


def decode(data):
    if len(data)<8:raise ValueError('Truncated native determinant header')
    count,n=struct.unpack_from('<II',data)
    if not 1<=n<=64 or count<1 or len(data)!=8+count*(8+n):raise ValueError('Native output length/header mismatch')
    pairs=[];seen=set()
    for k in range(count):
        start=8+k*(8+n);value=struct.unpack_from('<d',data,start)[0];a=b=0
        if not math.isfinite(value):raise ValueError('Nonfinite native CI amplitude')
        for i,char in enumerate(data[start+8:start+8+n]):
            if char not in b'02ab':raise ValueError('Invalid native occupancy byte')
            if char in b'2a':a|=1<<i
            if char in b'2b':b|=1<<i
        if (a,b) in seen:raise ValueError('Duplicate native determinant pair')
        seen.add((a,b));pairs.append([a,b,value])
    return n,pairs


def definitions():
    def ferro(k):
        h=np.zeros((2,2));g=np.zeros((2,2,2,2));g[0,0,0,0]=g[1,1,1,1]=.8 if k<.1 else 1.2
        g[0,0,1,1]=g[1,1,0,0]=.1
        for i,j,l,m in [(0,1,0,1),(0,1,1,0),(1,0,0,1),(1,0,1,0)]:g[i,j,l,m]=k
        return h,g
    h,g=ferro(.04);h=np.array([[-1.,.11],[.11,-.2]])
    rng=np.random.default_rng(20261025);a=rng.normal(size=(4,4))*.03
    hole_h=(a+a.T)/2+np.diag([-1.2,-.7,-.3,.2]);factors=rng.normal(size=(3,4,4))*.08
    factors=(factors+factors.transpose(0,2,1))/2;hole_g=np.einsum('lij,lkm->ijkm',factors,factors)
    common=dict(nelec=[1,1],core_energy=.37,spin_keyword=0,initial_strings=[1,2],expansion=False)
    rows=[dict(common,name='GENERAL_FULL',h=h,g=g),
        dict(common,name='FERRO_WEAK_SPIN0',h=ferro(.05)[0],g=ferro(.05)[1]),
        dict(common,name='RESTRICTED_SINGLE_PAIR',h=h,g=g,initial_strings=[2]),
        dict(common,name='HOLE_THREE_OF_FOUR',h=hole_h,g=hole_g,nelec=[3,3],initial_strings=[7,11,13,14]),
        dict(common,name='FERRO_STRONG_SPIN0',h=ferro(.3)[0],g=ferro(.3)[1]),
        dict(common,name='FERRO_WEAK_NO_SPIN_KEYWORD',h=ferro(.05)[0],g=ferro(.05)[1],spin_keyword=None),
        dict(common,name='NATIVE_HCI_EXPANSION',h=h,g=g,initial_strings=[1],expansion=True)]
    return rows


def exact_matrices(h,g,ne,core):
    from pyscf import fci
    n=len(h);strings=list(map(int,fci.cistring.make_strings(range(n),ne[0])));d=len(strings)**2
    h2=fci.direct_spin1.absorb_h1e(h,g,n,tuple(ne),.5)
    H=np.empty((d,d));S=np.empty_like(H)
    for i in range(d):
        v=np.zeros((len(strings),len(strings)));v.flat[i]=1
        H[:,i]=fci.direct_spin1.contract_2e(h2,v,n,tuple(ne)).ravel()
        S[:,i]=fci.spin_op.contract_ss(v,n,tuple(ne)).ravel()
    H+=core*np.eye(d)
    if np.max(abs(H-H.T))>1e-12 or np.max(abs(S-S.T))>1e-12:raise RuntimeError('Independent toy matrices not Hermitian')
    return strings,H,S


def prepare():
    from pyscf import tools
    RUN.mkdir(exist_ok=False);rows=[]
    for definition in definitions():
        name=definition['name'];folder=RUN/'inputs'/name;folder.mkdir(parents=True)
        h,g,ne=definition['h'],definition['g'],definition['nelec'];n=len(h);initial=definition['initial_strings']
        strings,H,S=exact_matrices(h,g,ne,definition['core_energy'])
        reference_strings=strings if definition['expansion'] else initial
        indices=[strings.index(a)*len(strings)+strings.index(b) for a in reference_strings for b in reference_strings]
        hp,sp=H[np.ix_(indices,indices)],S[np.ix_(indices,indices)]
        values,vecs=np.linalg.eigh(hp);penvalues,penvecs=np.linalg.eigh(hp+.1*sp)
        svals,svecs=np.linalg.eigh(sp);singlet=svecs[:,abs(svals)<1e-8]
        e0=None if not singlet.shape[1] else float(np.linalg.eigvalsh(singlet.T@hp@singlet)[0])
        tools.fcidump.from_integrals(folder/'fcidump.txt',h,g,n,tuple(ne),nuc=definition['core_energy'])
        for spin in ['Alpha','Beta']:write(folder/(spin+'Dets.bin'),b''.join(int(s).to_bytes(16,'big') for s in initial))
        lines=['nroots 1']+([] if definition['spin_keyword'] is None else ['spin '+str(definition['spin_keyword'])])
        lines+=['orbitals fcidump.txt','schedule','0 '+('1e-12' if definition['expansion'] else '2147483647'),'end',
            'davidsonTol 1e-10','dE 1e-10','maxiter '+('2' if definition['expansion'] else '1'),'noio',
            'writeBestDeterminants '+str(len(strings)**2),'nPTiter 0','DoSpinRDM','DoRDM','DoSpinOneRDM','DoOneRDM',
            'nocc '+str(sum(ne)),' '.join(map(str,range(sum(ne)))),'end']
        write(folder/'input.dat',('\n'.join(lines)+'\n').encode())
        row={k:v for k,v in definition.items() if k not in ['h','g']}
        row.update(norb=n,all_strings=strings,H_full=H.tolist(),S2_full=S.tolist(),
            free_minimum_energy_Ha=float(values[0]),singlet_minimum_energy_Ha=e0,
            finite_penalty_physical_energy_Ha=float(penvecs[:,0]@hp@penvecs[:,0]),
            finite_penalty_objective_Ha=float(penvalues[0]),
            input_SHA256={p.name:sha256(p.read_bytes()) for p in sorted(folder.iterdir())})
        rows.append(row)
    save(RUN/'fixtures.json',dict(version='N2_SYNTHETIC_NATIVE_DICE_PHYSICS_CONTRACT_V1',fixtures=rows,
        implementation_SHA256=sha256(Path(__file__).read_bytes()),energy_tolerance_Ha=1e-8,
        residual_tolerance=1e-8,spin_RDM_tolerance=1e-8,production_molecular_outcomes=0,
        purpose='DETERMINE_NATIVE_SPIN_SEMANTICS_NOT_ASSUME_S0_OR_FINITE_PENALTY_PARITY',new_model_shots=0))
    print('FROZEN_SYNTHETIC_INPUTS',len(rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare']);args=parser.parse_args();prepare()
