"""Small, independent *saved-basis* CI check; no new sampling or FCI reference.

The selected space has at most 16 determinants. The original small-cohort
OpenFermion-big-endian / blocked-fermion phase convention is explicit here.
"""
import json
import time
from pathlib import Path
from .evidence import sha256_file, write_json


def run(repo:Path,output:Path):
    import numpy as np
    import scipy.sparse as sp
    from scipy.linalg import eigh
    from threadpoolctl import threadpool_limits
    fixtures=repo/"data/fixtures"
    reference=json.loads((fixtures/"reference.json").read_text())
    if reference["n_qubits"]!=10 or reference["case"]["molecule"]!="LiH":
        raise ValueError("Only the bounded LiH 1.5-A saved fixture is supported")
    full=sp.load_npz(fixtures/"hamiltonian.npz")
    records=json.loads((fixtures/"lih15-saved-bases.json").read_text())
    n=reference["n_orbitals"]
    if reference["nalpha"]!=reference["nbeta"]:raise ValueError("Equal spin populations required")
    def act(x,q,create):
        if bool(x>>q&1)==create:return None
        return x^(1<<q),(-1)**((x&((1<<q)-1)).bit_count())
    def index(x):
        return sum(((x>>i)&1)<<(2*n-1-2*i)for i in range(n))+sum(((x>>(n+i))&1)<<(2*n-2-2*i)for i in range(n))
    values=[];cpu=time.process_time()
    with threadpool_limits(limits=1):
        for row in records:
            strings=json.loads(row["selected_spin_strings"])
            if len(strings)>4 or any(x.bit_count()!=reference["nalpha"]for x in strings):raise ValueError("Fixture cap/sector")
            bits=[a|(b<<n)for a in strings for b in strings]
            lookup={x:i for i,x in enumerate(bits)}
            s2=np.zeros((len(bits),len(bits)))
            # M_S=0: S^2 = S+ S-. Apply annihilation/creation right to left.
            for j,x in enumerate(bits):
                for q in range(n):
                    a=act(x,q,False)
                    if a is None:continue
                    b=act(a[0],n+q,True)
                    if b is None:continue
                    for p in range(n):
                        c=act(b[0],n+p,False)
                        if c is None:continue
                        d=act(c[0],p,True)
                        if d is not None and d[0]in lookup:s2[lookup[d[0]],j]+=a[1]*b[1]*c[1]*d[1]
            se,sv=eigh(s2)
            transform=sv[:,np.abs(se)<=1e-10]
            if not transform.shape[1]:raise ValueError("No exact singlet")
            signs=np.array([(-1)**sum(j<i for i in range(n)if a>>i&1 for j in range(n)if b>>j&1)for a in strings for b in strings])
            raw=[index(x)for x in bits]
            matrix=full[raw][:,raw].toarray()*signs[:,None]*signs[None,:]
            reduced=transform.T@matrix@transform
            eig,vec=eigh(reduced,subset_by_index=[0,0])
            energy=float(eig[0]);residual=float(np.linalg.norm(reduced@vec[:,0]-energy*vec[:,0]))
            delta=abs(energy-float(row["energy_Ha"]))
            if delta>1e-8 or residual>1e-8:raise ValueError("Saved CI parity failed")
            values.append({"selected_spin_strings":strings,"D":len(bits),"singlet_dimension":transform.shape[1],
                           "energy_Ha":energy,"saved_energy_Ha":float(row["energy_Ha"]),"absolute_error_Ha":delta,"residual_norm":residual})
    result={"passed":True,"scope":"two saved LiH bases only; not full source regeneration or full-cohort CI verification",
            "new_model_samples":0,"new_QPU_shots":0,"new_FCI_reference_computations":0,
            "saved_basis_CI_checks":len(values),"CPU_seconds":time.process_time()-cpu,"checks":values,
            "input_sha256":{p.name:sha256_file(p)for p in fixtures.iterdir()if p.is_file()}}
    if output.resolve().is_relative_to((repo/"data").resolve()):raise ValueError("Do not overwrite evidence")
    write_json(output,result)
    return result
