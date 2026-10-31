"""PySCF raw-pointer RDM adapter: force contiguous values, not new physics."""
from time import process_time
import numpy as np
from pyscf.fci import selected_ci
from .portable_penalized_ci import expectation


def audit(h,g,core,ne,strings,v,result):
    cpu=process_time();s=np.asarray(strings,np.int64)
    if np.iscomplexobj(v) or not np.isfinite(v).all():raise ValueError('Registered real finite root required')
    contiguous=np.array(v.reshape(len(s),len(s)),dtype=np.float64,order='C',copy=True)
    ci=selected_ci._as_SCIvector(contiguous,(s,s));solver=selected_ci.SelectedCI();solver.verbose=0
    one,two=solver.make_rdm12(ci,len(h),ne);energy=expectation(h,g,core,one,two);spin=float(solver.spin_square(ci,len(h),ne)[0])
    number=sum(ne);errors=dict(RDM_energy_Ha=abs(energy-result['energy_Ha']),S2=abs(spin-result['spin_square']),
        particle_number=abs(float(np.trace(one))-number),pair_number=abs(float(np.einsum('pprr->',two))-number*(number-1)))
    if max(errors.values())>1e-7:raise ValueError('Independent contiguous RDM/sector mismatch '+str(errors))
    return dict(errors=errors,CPU_seconds=process_time()-cpu,input_C_contiguous=bool(v.flags.c_contiguous),
        explicit_contiguous_copy=True,source_values_preserved=np.array_equal(contiguous.ravel(),v))
