"""Bounded explicit arbitrary-pair CI; never overwrites C7 or native DICE."""
import ctypes,hashlib,json,math,subprocess,sys,shutil
from pathlib import Path
from time import perf_counter,process_time
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import LinearOperator,eigsh

SOURCE=Path(__file__).with_suffix('.c')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_library(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    target=folder/('libjoint_ci.dylib' if sys.platform=='darwin' else 'libjoint_ci.so')
    manifest=folder/'build.json'
    if manifest.exists():
        record=json.loads(manifest.read_text())
        if record['source_SHA256']!=sha(SOURCE) or record['binary_SHA256']!=sha(target):raise RuntimeError('Private library drift')
        return target
    if target.exists():raise RuntimeError('Unreconciled existing private library')
    prefix=['rtk','proxy'] if shutil.which('rtk') else []
    command=prefix+['cc','-O3','-std=c11','-D_POSIX_C_SOURCE=200809L','-shared','-fPIC',str(SOURCE),'-o',str(target)]
    start=perf_counter();p=subprocess.run(command,capture_output=True,text=True,timeout=45)
    if p.returncode:raise RuntimeError('Private sparse-CI build failed: '+p.stderr)
    record=dict(source_SHA256=sha(SOURCE),binary_SHA256=sha(target),command=command,
        compiler=subprocess.run(prefix+['cc','--version'],capture_output=True,text=True,timeout=10).stdout,
        build_wall_seconds=perf_counter()-start)
    # Build metadata only; scientific artifacts are written by the caller.
    manifest.write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
    return target


def matrices(h,g,nelec,pairs,library,seconds=60,triangle_cap=8000000):
    start,wall=process_time(),perf_counter();h=np.asarray(h);g=np.asarray(g);n=len(h);raw=np.asarray(pairs)
    if not 1<=n<=32 or h.shape!=(n,n) or g.shape!=(n,)*4 or np.iscomplexobj(h) or np.iscomplexobj(g):
        raise ValueError('Qualified real spatial-integral n<=32 scope only')
    if raw.ndim!=2 or raw.shape[1]!=2 or not np.issubdtype(raw.dtype,np.integer) or raw.dtype==np.bool_:
        raise ValueError('Explicit integer alpha/beta pairs required')
    if len(raw)<1 or len(raw)>32768 or len({tuple(map(int,x)) for x in raw})!=len(raw):raise ValueError('Bounded unique joint basis required')
    if any(int(x)<0 or int(x)>=1<<n or int(x).bit_count()!=nelec[s] for row in raw for s,x in enumerate(row)):
        raise ValueError('Joint physical particle-sector mismatch')
    if not np.isfinite(h).all() or not np.isfinite(g).all():raise ValueError('Nonfinite integrals')
    if max(np.max(abs(h-h.T)),np.max(abs(g-g.transpose(1,0,2,3))),np.max(abs(g-g.transpose(2,3,0,1))))>1e-10:
        raise ValueError('Real Hermitian integral symmetries required')
    h=np.ascontiguousarray(h,dtype=np.float64);g=np.ascontiguousarray(g,dtype=np.float64)
    pairs=np.ascontiguousarray(raw,dtype=np.uint64);d=len(pairs);cap=min(d*(d+1)//2,int(triangle_cap))
    if cap<1 or seconds<=0:raise ValueError('Positive resource bounds required')
    rows=np.empty(cap,np.int32);cols=np.empty(cap,np.int32);hv=np.empty(cap);sv=np.empty(cap)
    lib=ctypes.CDLL(str(Path(library).resolve()));fn=lib.joint_matrices;fn.restype=ctypes.c_int64
    ptr=ctypes.c_void_p
    fn.argtypes=[ptr,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_int,ptr,ctypes.c_int,ctypes.c_int64,ctypes.c_double,ptr,ptr,ptr,ptr]
    count=fn(h.ctypes.data,g.ctypes.data,n,int(nelec[0]),int(nelec[1]),pairs.ctypes.data,d,cap,float(seconds),
        rows.ctypes.data,cols.ctypes.data,hv.ctypes.data,sv.ctypes.data)
    if count==-1:raise MemoryError('Bounded explicit CI triangle capacity exceeded; no partial root')
    if count==-2:raise TimeoutError('Bounded explicit CI matrix construction wall cap; no partial root')
    if not 0<count<=cap:raise RuntimeError('Invalid native sparse-builder return')
    def symmetric(values):
        m=coo_matrix((values[:count],(rows[:count],cols[:count])),shape=(d,d)).tocsr()
        m=m+m.T-diags(m.diagonal());m.eliminate_zeros();m.sort_indices();return m
    H,S=symmetric(hv),symmetric(sv)
    return H,S,dict(matrix_build_CPU_seconds=process_time()-start,matrix_build_wall_seconds=perf_counter()-wall,
        triangle_entries=count,triangle_capacity=cap,H_nonzeros=H.nnz,S2_nonzeros=S.nnz,
        CSR_bytes=sum(x.data.nbytes+x.indices.nbytes+x.indptr.nbytes for x in [H,S]),
        matrix_element_threshold=0.,binary_SHA256=sha(library),source_SHA256=sha(SOURCE))


def solve(h,g,core,nelec,pairs,library,seconds=120,triangle_cap=8000000):
    cpu,wall=process_time(),perf_counter();H,S,info=matrices(h,g,nelec,pairs,library,seconds=seconds,triangle_cap=triangle_cap)
    A=H+.1*S;d=A.shape[0];calls=0
    def hop(x):
        nonlocal calls
        if perf_counter()-wall>seconds:raise TimeoutError('Bounded explicit joint-CI solve wall cap')
        calls+=1;return A@x
    rng=np.random.default_rng(20261025)
    # Match C7's three random bilinear probes and then its random v0 draw.
    errors=[]
    for _ in range(3):
        x,y=rng.normal(size=(2,d));x/=np.linalg.norm(x);y/=np.linalg.norm(y)
        errors.append(float(abs(x@hop(y)-hop(x)@y)))
    v0=rng.normal(size=d);v0/=np.linalg.norm(v0)
    if d<=2:values,vectors=np.linalg.eigh(A.toarray())
    else:values,vectors=eigsh(LinearOperator((d,d),matvec=hop,dtype=np.float64),k=2,which='SA',v0=v0,tol=1e-12,maxiter=1000)
    order=np.argsort(values);values=values[order];v=vectors[:,order[0]]
    residual=float(np.linalg.norm(hop(v)-values[0]*v));energy=float(v@(H@v)+core);s2=float(v@(S@v))
    objective=float(values[0]+core)
    if residual>1e-8 or max(errors)>1e-10 or abs(objective-energy-.1*s2)>1e-8 or abs(v@v-1)>1e-10:
        raise RuntimeError('Explicit joint-CI physical/spin/residual consistency failure')
    return dict(energy_Ha=energy,penalized_energy_Ha=objective,spin_square=s2,residual_norm=residual,
        actual_CI_dimension=d,matvec_calls=calls,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
        lowest_two_eigenvalues_Ha=(values[:2]+core).tolist(),random_bilinear_hermiticity_errors=errors,
        common_equation='PROJECTED_PHYSICAL_H_PLUS_0p1_S2_NOT_EXACT_S0_NO_OO',
        independent_RDM_qualification='SEPARATE_PYSCF_TESTS_NOT_PER_PRODUCTION_ROOT',**info),v
