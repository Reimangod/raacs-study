"""Read public P3 evidence; never merge different Hamiltonians by label."""
from io import BytesIO
import json
from pathlib import Path
import urllib.request
import zipfile
import numpy as np
import ffsim
from .common_execution import load_physical
from .input_adapter import array_bytes
from .io import ROOT,atomic_bytes,save_json,sha256
from .lucj import interaction_pairs,make_circuit,circuit_views

COMMIT="865bd762d3f96ee79bb362eb3dd46951b48b426e"
BASE="https://raw.githubusercontent.com/jrm874/sqd_data_repository/"+COMMIT+"/"
PATHS=["numerics/2Fe-2S_optimal_circuits/active_space_integrals/h1e_Fe2S2_MO.npy",
       "numerics/2Fe-2S_optimal_circuits/active_space_integrals/h2e_Fe2S2_MO.npy",
       "numerics/2Fe-2S_optimal_circuits/active_space_integrals/run_RHF.py",
       "numerics/2Fe-2S_optimal_circuits/circuit_parameters/optimal_params.npy",
       "numerics/2Fe-2S_optimal_circuits/circuit_parameters/load_params_to_circuit.py",
       "classical_reference_energies/2Fe-2S/classical_methods_energies.txt",
       "experiments/N2_6-31G/experiment_data.zip","experiments/2Fe-2S/experiment_data.zip"]


def run():
    root=ROOT/"sources/P3-public";records=[]
    for name in PATHS:
        path=root/name
        if not path.exists():
            with urllib.request.urlopen(BASE+name,timeout=60) as stream:blob=stream.read(40*2**20)
            if len(blob)>=40*2**20:raise RuntimeError("Source exceeds registered single-file download bound")
            atomic_bytes(path,blob)
        blob=path.read_bytes();row={"path":name,"sha256":sha256(blob),"bytes":len(blob),"url":BASE+name}
        if name.endswith(".zip"):
            with zipfile.ZipFile(BytesIO(blob)) as z:
                row["archive_members"]=[{"name":x.filename,"bytes":x.file_size} for x in z.infolist()]
            # Inspect members before parsing; do not execute pickle or notebook code.
        records.append(row)
    save_json(root/"source-manifest.json",{"repository":"jrm874/sqd_data_repository","commit":COMMIT,"files":records,
              "downloaded_artifacts_executed":False})
    prefix=root/"numerics/2Fe-2S_optimal_circuits"
    h1=np.load(prefix/"active_space_integrals/h1e_Fe2S2_MO.npy",allow_pickle=False)
    eri=np.load(prefix/"active_space_integrals/h2e_Fe2S2_MO.npy",allow_pickle=False)
    m,h,g=load_physical("FE2S2-40-P2")
    if h1.shape!=h.shape or eri.shape!=g.shape:raise RuntimeError("Different active spaces")
    comparison={"case_id":"FE2S2-40-P2","C_lane_problem_id":m["problem_id"],
                "max_h1_difference":float(np.max(abs(h1-h))),"max_eri_difference":float(np.max(abs(eri-g))),
                "P3_core_value_verified":False,"same_Hamiltonian_established":False,
                "reason":"Core and orbital-gauge equivalence require explicit verification; labels or dimensions are insufficient"}
    save_json(root/"Fe-integral-crosswalk.json",comparison)
    params=np.load(prefix/"circuit_parameters/optimal_params.npy",allow_pickle=False)
    print("PARAMETERS",params.shape,"INTEGRAL_CROSSWALK",comparison,flush=True)
    for row in records:
        if "archive_members"in row:print(row["path"],json.dumps(row["archive_members"]),flush=True)
    return records


if __name__=="__main__":run()
