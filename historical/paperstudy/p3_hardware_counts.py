"""Safely inspect source NumPy object archives without arbitrary unpickling.

Only the three NumPy constructors present in the pinned bytes are allowed.
The records remain H-lane evidence, never ideal-sampler common-budget cells.
"""
from collections import Counter
from io import BytesIO
import json
import pickle
import zipfile
import numpy as np
from .input_adapter import array_bytes
from .io import ROOT,atomic_bytes,save_json,sha256


class NumpyOnly(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={("numpy","ndarray"):np.ndarray,("numpy","dtype"):np.dtype,
                 ("numpy.core.multiarray","_reconstruct"):np._core.multiarray._reconstruct,
                 ("numpy._core.multiarray","_reconstruct"):np._core.multiarray._reconstruct}
        if (module,name) not in allowed:raise pickle.UnpicklingError("Unapproved global: "+module+"."+name)
        return allowed[module,name]


def load(case):
    filename=ROOT/"sources/P3-public/experiments"/case/"experiment_data.zip"
    with zipfile.ZipFile(filename) as z:
        name=next(x for x in z.namelist() if x.startswith("experiment_data/") and x.endswith(".npy"))
        info=z.getinfo(name)
        if info.file_size>160*2**20:raise MemoryError("Registered archive inspection bound")
        raw=z.read(name)
    stream=BytesIO(raw);version=np.lib.format.read_magic(stream)
    shape,order,dtype=(np.lib.format.read_array_header_1_0(stream) if version==(1,0) else np.lib.format.read_array_header_2_0(stream))
    if not dtype.hasobject or len(shape)!=1 or shape[0]>1000:raise ValueError("Unexpected hardware archive structure")
    value=NumpyOnly(stream).load()
    if not isinstance(value,np.ndarray) or value.shape!=shape:raise ValueError("Archive/header shape mismatch")
    for item in value:
        if not isinstance(item,dict):raise ValueError("Expected dictionaries, not executable objects")
        if any(not isinstance(k,str) or not set(k)<=set("01 ") or not isinstance(v,(int,float,np.integer,np.floating)) or isinstance(v,(bool,np.bool_)) or not np.isfinite(v) for k,v in item.items()):
            raise ValueError("Invalid bitstring count dictionary")
    return value,sha256(raw)


def run():
    root=ROOT/"hardware-reanalysis-v1";rows=[]
    for case in ("N2_6-31G","2Fe-2S"):
        data,raw_sha=load(case)
        groups=[(str(round(.7+.1*i,1)),data[2*i:2*i+2]) for i in range(24)] if case=="N2_6-31G" else [("cluster",data)]
        if case=="N2_6-31G" and len(data)!=48:raise RuntimeError("Source helper grouping changed")
        for label,dictionaries in groups:
            merged=Counter()
            for d in dictionaries:
                for key,v in d.items():merged[key.replace(" ","")]+=float(v)
            widths=sorted(set(map(len,merged)))
            if len(widths)!=1:raise RuntimeError("Mixed register widths")
            bits=widths[0];n=bits//2
            ne=5 if case=="N2_6-31G" else 15
            valid=sum(v for k,v in merged.items() if (int(k,2)&((1<<n)-1)).bit_count()==ne and (int(k,2)>>n).bit_count()==ne)
            mass=sum(merged.values())
            if not np.isfinite(mass) or mass<=0:raise ValueError("Invalid distribution mass")
            row={"lane":"H_HARDWARE_WEIGHTED_DISTRIBUTION_REANALYSIS","source_case":case,"source_helper_label":label,
                 "total_hardware_shots":None,"shots_unknown_reason":"Archive values are floating weights/probabilities, not integer event multiplicities",
                 "source_value_semantics":"Floating signed weights as distributed; readout-mitigation provenance not independently closed",
                 "total_weight_mass":mass,"negative_weight_mass":sum(v for v in merged.values() if v<0),
                 "observed_unique":len(merged),"register_width":bits,
                 "valid_Nalpha_Nbeta_signed_weight_fraction_under_blocked_JW":valid/mass,
                 "spin_purity_proved":False,"configuration_recovery_performed":False,
                 "source_array_sha256":raw_sha,"directly_comparable_C_lane_problem":False,
                 "Hamiltonian_binding":"P3 source-specific integral/core crosswalk required; do not merge with P2 by molecule name",
                 "distance_unit":"Source load helper label only; physical geometry not independently bound here" if case=="N2_6-31G" else "Cluster, not diatomic bond scan"}
            path=root/case/label
            ids=np.array([int(k,2) for k in sorted(merged)],np.uint64);weights=np.array([merged[k] for k in sorted(merged)],np.float64)
            atomic_bytes(path/"hardware-distribution.npz",array_bytes({"blocked_ids_source_order":ids,"source_weights":weights}))
            row["distribution_sha256"]=sha256((path/"hardware-distribution.npz").read_bytes())
            save_json(path/"count-summary.json",row);rows.append(row)
            print(case,label,"FLOAT_WEIGHTS_NOT_RAW_COUNTS",row["total_weight_mass"],row["valid_Nalpha_Nbeta_signed_weight_fraction_under_blocked_JW"],flush=True)
    save_json(root/"summary.json",{"rows":rows,"new_actual_QPU_shots":0,"ideal_model_shots_added":0,
              "state":"PUBLIC_COUNTS_INSPECTED_HAMILTONIAN_AND_ORIGINAL_POSTPROCESSING_NOT_YET_CLOSED",
              "safe_deserialization":"NumPy-only allowlist, no source code or unrestricted pickle execution"})


if __name__=="__main__":run()
