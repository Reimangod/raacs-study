"""Use the existing public SelectedCI initial-guess policy, not an oracle root."""
import numpy as np
from pyscf.fci import selected_ci


def native_component(data):
    s=np.asarray(data['strings'],np.int64);n=int(data['n']);k=int(data['k'])
    diagonal=np.asarray(data['exact_diagonal'],float)
    if diagonal.shape!=(len(s)**2,) or not np.all(np.isfinite(diagonal)):
        raise ValueError('Complete finite projected diagonal required')
    solver=selected_ci.SelectedCI();solver.verbose=0
    return np.asarray(solver.get_init_guess((s,s),n,(k,k),1,diagonal)[0],float).ravel().copy()


def initial_guess(data):
    """Keep a nonzero perturbation in both spin-exchange sectors."""
    base=native_component(data);random=np.asarray(data['initial_vector'],float)
    if random.shape!=base.shape or not np.all(np.isfinite(random)) or np.linalg.norm(random)==0:
        raise ValueError('Registered full-support perturbation required')
    return base+.001*random/np.linalg.norm(random)
