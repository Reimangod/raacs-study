"""P3-specific physical inputs: FCIDUMP core, no fitting to target energies."""
import json
import numpy as np
from pyscf import ao2mo
from pyscf.tools import fcidump
from .input_adapter import array_bytes,hf_energy
from .io import ROOT,atomic_bytes,save_json,digest,sha256


def run():
    source=ROOT/"sources/P3-public"
    rows=[]
    table=np.loadtxt(source/"classical_reference_energies/N2_6-31G/energies_n2_6-31g.txt",skiprows=2)
    for case,label,relative,n,ne in (("P3-N2-ratio1p00","1.0","integrals/N2_6-31G/R_16_1.00_fcidump.txt",16,5),
            ("P3-N2-ratio2p00","2.0","integrals/N2_6-31G/R_16_2.00_fcidump.txt",16,5),
            ("P3-FE2S2-40","cluster","integrals/2Fe-2S/fcidump_Fe2S2_MO.txt",20,15)):
        original=source/relative;raw=fcidump.read(str(original),verbose=False)
        if raw["NORB"]!=n or raw["NELEC"]!=2*ne or raw["MS2"]!=0:raise RuntimeError("P3 physical sector mismatch")
        h=raw["H1"];g=ao2mo.restore(1,raw["H2"],n);core=float(raw["ECORE"])
        EHF=hf_energy(h,g,core,ne,ne)
        payload=array_bytes({"h1":h,"eri":g})
        physical={"norb":n,"nelec":[ne,ne],"core_energy":core,"h1_sha256":sha256(h.astype("<f8").tobytes()),
                  "eri_sha256":sha256(g.astype("<f8").tobytes()),"mapping":"blocked-spin JW, orbital0 at least significant bit of each spin block"}
        meta={"case_id":case,"problem_id":digest(physical),**physical,"hf_energy_Ha":EHF,
              "source_fcidump":relative,"source_fcidump_sha256":sha256(original.read_bytes()),
              "core_from_FCIDUMP_not_energy_fit":True,"source_weight_label":label,
              "same_as_C_lane_problem_assumed":False,"circuit_parameters_source":"UNKNOWN for the original hardware execution"}
        if case.startswith("P3-N2"):
            ratio=float(label);reference=table[np.argmin(abs(table[:,0]-ratio))]
            if abs(reference[0]-ratio)>1e-8 or abs(EHF-reference[1])>1e-7:raise RuntimeError("P3 HF/table identity failed")
            meta.update(repository_distance_ratio=ratio,bond_angstrom=ratio*1.0975135,
                distance_evidence="Pinned repository README explicitly defines ratio units; independent RHF at ratio1 gave -108.86777367563818 Ha, matching source, whereas 1.0 Angstrom gave -108.8352365707745 Ha",
                manuscript_caption_unit_discrepancy_not_silently_resolved=True)
            ref={"reference_energy_Ha":float(reference[-1]),"kind":"AUTHOR_TABLE_FCI_ROUNDED_8_DECIMALS",
                 "absolute_precision_bound_certified":False,"reference_sector_independently_verified":False}
        else:
            ref={"reference_energy_Ha":-116.6056091,"kind":"AUTHOR_REPORTED_DMRG_APPROXIMATE",
                 "source":"classical_reference_energies/2Fe-2S/classical_methods_energies.txt",
                 "absolute_precision_bound_certified":False,"reference_sector_independently_verified":False}
        directory=ROOT/"hardware-physical-v1"/case
        atomic_bytes(directory/"integrals.npz",payload);meta["integrals_file_sha256"]=sha256(payload)
        save_json(directory/"physical.json",meta);save_json(directory/"evaluation-reference.json",ref)
        rows.append(meta)
    save_json(ROOT/"hardware-physical-v1/summary.json",{"rows":rows,"C_lane_equivalence_claimed":False,
              "requested_historical_SQD_version":"0.1.0","version_status":"Not available in public PyPI releases or GitHub tags checked; earliest public tag/release0.3.0",
              "new_reanalysis_runtime":"Current qualified qiskit-addon-sqd0.14.0, explicit non-historical reproduction"})
    print(json.dumps(rows),flush=True)


if __name__=="__main__":run()
