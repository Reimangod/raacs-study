from pathlib import Path
import numpy as np
from scipy import sparse
from .ansatz import make_pool,pool_catalog,hf_state,state_from_parameters,energy_and_gradient,selection_gradients
from .common import save_json,save_npz,ROOT


def main():
    root=ROOT/"results/engine-v2-qualification/all-pool-check";rng=np.random.default_rng(619)
    for method in ("ceo_ovp","ceo_mvp","qeb","qubit","gsd"):
        pool=make_pool(method,8);catalog=pool_catalog(pool);n=256
        # Fixed synthetic Hermitian H exercises complex amplitudes without FCI labels.
        a=sparse.random(n,n,density=.03,random_state=rng,format="csr")
        h=(a+a.T).tocsr();indices=rng.choice(pool.size,min(pool.size,12),replace=False).tolist()
        theta=rng.uniform(-1,1,len(indices));hf=240;initial=hf_state(8,hf)
        state=state_from_parameters(pool,indices,theta,initial);energy,gradient=energy_and_gradient(pool,indices,theta,initial,h)
        dest=root/method;save_json(dest/"pool.json",catalog)
        save_npz(dest/"expected.npz",h_data=h.data,h_indices=h.indices,h_indptr=h.indptr,indices=np.array(indices),
                 theta=theta,hf_id=np.array(hf),state=state,energy=np.array(energy),gradient=gradient,
                 pool_gradient=selection_gradients(pool,state,h))
    print(root)


if __name__=="__main__":main()
