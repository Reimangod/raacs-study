"""S5 worker: exact CuPy LUCJ with independent full ffsim state comparison.

Derived from gpu_lucj_engine.py; this is a new version, not a mutation of the
qualified worker. Large reference states are compared in RAM, never saved.

No ffsim/PySCF installation is needed on the GPU node. The host supplies the
exact uncompressed ffsim decomposition and the physical UCJ coefficients.
Only primitive operations are ported; initialization/ansatz are not changed.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
from time import perf_counter, process_time

import numpy as np


def spin_strings(n, electrons):
    return np.array(sorted(sum(1 << q for q in index) for index in itertools.combinations(range(n), electrons)), np.uint32)


def pairs(strings, i, j):
    first = np.flatnonzero(((strings >> i) & 1) & (1-((strings >> j) & 1)))
    second = np.searchsorted(strings, strings[first] ^ np.uint32((1 << i) | (1 << j)))
    if not np.array_equal(strings[second], strings[first] ^ np.uint32((1 << i) | (1 << j))):
        raise RuntimeError("Complete number-sector Givens pairing failed")
    return first.astype(np.int32), second.astype(np.int32)


def numpy_state(config, arrays):
    n, na, nb = config["norb"], *config["nelec"]
    sa, sb = spin_strings(n, na), spin_strings(n, nb)
    if len(sa)*len(sb) > 1000000:
        raise ValueError("CPU emulator is a bounded qualification, not the large execution route")
    wa = ((sa[:, None] >> np.arange(n)) & 1).astype(float)
    wb = ((sb[:, None] >> np.arange(n)) & 1).astype(float)
    state = np.zeros((len(sa), len(sb)), complex)
    state[0, 0] = 1
    for stage, rotation in enumerate(config["rotations"]):
        for c, sr, si, i, j in rotation["givens"]:
            s = sr+1j*si
            for spin, strings in enumerate((sa, sb)):
                first, second = pairs(strings, i, j)
                a = state[first, :].copy() if spin == 0 else state[:, first].copy()
                b = state[second, :].copy() if spin == 0 else state[:, second].copy()
                if spin == 0:
                    state[first, :] = c*a+s.conjugate()*b
                    state[second, :] = -s*a+c*b
                else:
                    state[:, first] = c*a+s.conjugate()*b
                    state[:, second] = -s*a+c*b
        D = np.array([complex(r, i) for r, i in rotation["diagonal"]])
        pha = np.prod(np.where(wa, D, 1), axis=1)
        phb = np.prod(np.where(wb, D, 1), axis=1)
        state *= pha[:, None]*phb[None, :]
        if stage < len(arrays["diag_coulomb_mats"]):
            jaa, jab = arrays["diag_coulomb_mats"][stage]
            aa = np.einsum("bi,ij,bj->b", wa, jaa, wa)/2
            bb = np.einsum("bi,ij,bj->b", wb, jaa, wb)/2
            state *= np.exp(1j*(aa[:, None]+bb[None, :]+wa@jab@wb.T))
    return state.reshape(-1)


CUDA = r'''
extern "C" __global__ void givens(double2* v, const int* first, const int* second,
 const long long da, const long long db, const int npairs, const int spin,
 const double c, const double sr, const double si) {
 long long k = (long long)blockDim.x*blockIdx.x+threadIdx.x;
 long long width = spin == 0 ? db : da;
 if(k >= (long long)npairs*width) return;
 int p = k/width; long long other = k%width;
 long long x = spin == 0 ? (long long)first[p]*db+other : other*db+first[p];
 long long y = spin == 0 ? (long long)second[p]*db+other : other*db+second[p];
 double2 a=v[x], b=v[y];
 v[x]=make_double2(c*a.x+sr*b.x+si*b.y,c*a.y+sr*b.y-si*b.x);
 v[y]=make_double2(-sr*a.x+si*a.y+c*b.x,-sr*a.y-si*a.x+c*b.y);
}
extern "C" __global__ void phases(double2* v,const double2* pa,const double2* pb,
 const long long da,const long long db) {
 long long k=(long long)blockDim.x*blockIdx.x+threadIdx.x;
 if(k>=da*db) return;
 double2 a=pa[k/db], b=pb[k%db], x=v[k];
 double r=a.x*b.x-a.y*b.y, s=a.x*b.y+a.y*b.x;
 v[k]=make_double2(r*x.x-s*x.y,r*x.y+s*x.x);
}
extern "C" __global__ void jastrow(double2* v,const unsigned int* sa,const unsigned int* sb,
 const double* aa,const double* bb,const int* ia,const int* ib,const double* coeff,
 const int terms,const long long da,const long long db) {
 long long k=(long long)blockDim.x*blockIdx.x+threadIdx.x;
 if(k>=da*db) return;
 long long a=k/db,b=k%db; double e=aa[a]+bb[b];
 for(int t=0;t<terms;t++) if(((sa[a]>>ia[t])&1)&&((sb[b]>>ib[t])&1)) e+=coeff[t];
 double cs=cos(e),sn=sin(e);double2 x=v[k];
 v[k]=make_double2(cs*x.x-sn*x.y,sn*x.x+cs*x.y);
}
'''


def execute(config_path, arrays_path, output):
    import cupy as cp
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start, cpu = perf_counter(), process_time()
    config = json.loads(Path(config_path).read_text())
    import hashlib
    expected_id = hashlib.sha256(json.dumps({k:v for k,v in config.items() if k!="input_id"},
        sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    if config["input_id"] != expected_id or hashlib.sha256(Path(arrays_path).read_bytes()).hexdigest() != config["arrays_sha256"]:
        raise RuntimeError("S5 configuration/arrays digest mismatch")
    with np.load(arrays_path, allow_pickle=False) as z:
        arrays = {key: z[key].copy() for key in z.files}
    independent_CPU_cost = None
    if config.get("independent_CPU_operator"):
        import ffsim
        import hashlib
        spec = config["independent_CPU_operator"]
        blob = Path(spec["path"]).read_bytes()
        if hashlib.sha256(blob).hexdigest() != spec["sha256"]:
            raise RuntimeError("Independent CPU operator mismatch")
        with np.load(spec["path"],allow_pickle=False) as z:
            op = ffsim.UCJOpSpinBalanced(z["diag_coulomb_mats"],z["orbital_rotations"],z["final_orbital_rotation"])
        tt,cc = perf_counter(),process_time()
        arrays["expected_sector_vector"] = ffsim.apply_unitary(
            ffsim.hartree_fock_state(config["norb"],tuple(config["nelec"])),
            op,config["norb"],tuple(config["nelec"]),copy=False)
        independent_CPU_cost = {"wall_seconds":perf_counter()-tt,"CPU_seconds":process_time()-cc,
                                "statevector_saved":False,"purpose":"independent exact implementation parity"}
    n = config["norb"]
    na, nb = config["nelec"]
    sa, sb = spin_strings(n, na), spin_strings(n, nb)
    da, db = len(sa), len(sb)
    if 16*da*db > 12*2**30:
        raise MemoryError("Registered single sector-vector engine limit12GiB")
    wa = ((sa[:, None] >> np.arange(n)) & 1).astype(float)
    wb = ((sb[:, None] >> np.arange(n)) & 1).astype(float)
    sa_gpu, sb_gpu = cp.asarray(sa), cp.asarray(sb)
    state = cp.zeros((da, db), cp.complex128)
    state[0, 0] = 1
    kernels = {name: cp.RawKernel(CUDA, name, options=("--std=c++11",)) for name in ("givens", "phases", "jastrow")}
    mappings = {}
    grid = ((da*db+255)//256,)
    for stage, rotation in enumerate(config["rotations"]):
        for c, sr, si, i, j in rotation["givens"]:
            if abs(i-j) != 1:
                raise ValueError("Only exact adjacent Givens primitives are qualified")
            for spin, strings in enumerate((sa, sb)):
                key = (spin, i, j)
                if key not in mappings:
                    first, second = pairs(strings, i, j)
                    mappings[key] = cp.asarray(first), cp.asarray(second)
                first, second = mappings[key]
                width = db if spin == 0 else da
                kernels["givens"](((len(first)*width+255)//256,), (256,),
                        (state, first, second, np.int64(da), np.int64(db), np.int32(len(first)), np.int32(spin),
                         np.float64(c), np.float64(sr), np.float64(si)))
        D = np.array([complex(r, i) for r, i in rotation["diagonal"]])
        pha = cp.asarray(np.prod(np.where(wa, D, 1), axis=1))
        phb = cp.asarray(np.prod(np.where(wb, D, 1), axis=1))
        kernels["phases"](grid, (256,), (state, pha, phb, np.int64(da), np.int64(db)))
        if stage < len(arrays["diag_coulomb_mats"]):
            jaa, jab = arrays["diag_coulomb_mats"][stage]
            aa = cp.asarray(np.einsum("bi,ij,bj->b", wa, jaa, wa)/2)
            bb = cp.asarray(np.einsum("bi,ij,bj->b", wb, jaa, wb)/2)
            ia, ib = np.nonzero(jab)
            kernels["jastrow"](grid, (256,), (state, sa_gpu, sb_gpu, aa, bb,
                  cp.asarray(ia, dtype=cp.int32), cp.asarray(ib, dtype=cp.int32), cp.asarray(jab[ia, ib]),
                  np.int32(len(ia)), np.int64(da), np.int64(db)))
    cp.cuda.Stream.null.synchronize()
    preparation_seconds = perf_counter()-start
    norm = float(cp.vdot(state, state).real.item())
    if abs(norm-1) > 1e-8:
        raise RuntimeError("Large number-sector GPU state norm failed")
    verification = None
    if "expected_sector_vector" in arrays:
        expected = cp.asarray(arrays.pop("expected_sector_vector")).reshape(state.shape)
        overlap = cp.vdot(expected, state)
        aligned = state*cp.exp(-1j*cp.angle(overlap))
        verification = {"state_max_error": float(cp.max(cp.abs(aligned-expected)).item()),
                        "probability_L1_error": float(cp.sum(cp.abs(cp.abs(state)**2-cp.abs(expected)**2)).item())}
        if max(verification.values()) > 1e-8:
            raise RuntimeError("Full-sector independent ffsim/GPU parity failed")
        del expected, aligned
    # State is retained on the GPU only; no large statevector output.
    cdf = cp.cumsum(cp.abs(state.reshape(-1))**2)
    cdf /= cdf[-1]
    results = []
    for request in config.get("sampling_requests", []):
        rng = cp.random.RandomState(request["seed"])
        sampled = cp.searchsorted(cdf, rng.random_sample(request["shots"]))
        unique, counts = cp.unique(sampled, return_counts=True)
        indices, counts = cp.asnumpy(unique), cp.asnumpy(counts)
        ids = sa[indices//db].astype(np.uint64) | (sb[indices % db].astype(np.uint64) << n)
        name = request["id"]+".npz"
        temporary = output/(name+".pending")
        with temporary.open("wb") as file:
            np.savez_compressed(file, blocked_ids=ids, counts=counts.astype(np.uint64))
        temporary.replace(output/name)
        results.append({**request, "count_file": name, "actual_model_shots": int(counts.sum())})
    cp.cuda.Stream.null.synchronize()
    result = {"status": "EXACT_SECTOR_GPU_LUCJ_COMPLETED", "input_id": config["input_id"],
              "norb": n, "nelec": [na, nb], "sector_dimension": da*db, "norm_squared": norm,
              "independent_ffsim_parity": verification, "independent_CPU_cost": independent_CPU_cost, "sampling_results": results,
              "GPU_preparation_seconds_including_CUDA_startup": preparation_seconds-(independent_CPU_cost["wall_seconds"] if independent_CPU_cost else 0),
              "wall_seconds": perf_counter()-start, "CPU_process_seconds": process_time()-cpu,
              "GPU_memory_pool_retained_bytes": int(cp.get_default_memory_pool().total_bytes()),
              "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), "cupy": cp.__version__,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "actual_QPU_shots": 0,
              "full_Fock_vector_allocated": False, "number_sector_vector_saved": False}
    temporary = output/"result.pending"
    temporary.write_text(json.dumps(result, indent=2)+"\n")
    temporary.replace(output/"result.json")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("arrays")
    parser.add_argument("output")
    args = parser.parse_args()
    execute(args.config, args.arrays, args.output)
