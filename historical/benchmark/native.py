"""Derived native resource estimates under an explicit SYNTHETIC target contract.

No physical QPU target/credentials were provided. This preset is not an IBM device
calibration or a hardware execution. It supplements, not replaces, logical resources.
"""
import argparse
import json
from pathlib import Path
import qiskit
from qiskit import qpy,transpile
from qiskit.transpiler import CouplingMap
from .common import save_json,file_digest,digest,resolve_artifact


def compile_native(qc):
    return transpile(qc,basis_gates=["rz","sx","x","ecr"],
                     coupling_map=CouplingMap.from_line(qc.num_qubits,bidirectional=True),
                     optimization_level=3,seed_transpiler=8137)


def native_resources(qc):
    native=compile_native(qc)
    return {"native_2q_count":sum(item.operation.num_qubits==2 for item in native.data),
            "native_2q_depth":native.depth(filter_function=lambda item:item.operation.num_qubits==2),
            "native_total_depth":native.depth(),"gate_counts":dict(native.count_ops()),
            "native_gate_family":"ECR","qiskit_version":qiskit.__version__,
            "target":"SYNTHETIC_BIDIRECTIONAL_ECR_LINE_NOT_PHYSICAL_QPU",
            "basis_gates":["rz","sx","x","ecr"],"optimization_level":3,"seed_transpiler":8137,
            "hardware_execution":False}


def run_native(root):
    root=Path(root);rows=[]
    for cp_path in sorted((root/"checkpoints").glob("*/*/step-*.json")):
        cp=json.loads(cp_path.read_text())
        circuit_path=resolve_artifact(root,cp.get("sampling_circuit_qpy",cp["circuit_qpy"]))
        expected=cp.get("sampling_circuit_sha256",cp["circuit_sha256"])
        if file_digest(circuit_path)!=expected:
            raise RuntimeError("Native compiler input circuit hash drift")
        contract={"schema":"native-ecr-line-1","checkpoint_id":cp["checkpoint_id"],
                  "input_qpy_sha256":expected,"qiskit_version":qiskit.__version__,"seed":8137}
        out=root/"native_resources/ecr-line"/cp["case_id"]/cp["method"]/cp_path.name
        if out.exists():
            row=json.loads(out.read_text())
            if row["context_id"]!=digest(contract):
                raise RuntimeError("Native compiler contract changed")
        else:
            with open(circuit_path,"rb") as f:
                qc=qpy.load(f)[0]
            row={"case_id":cp["case_id"],"method":cp["method"],"step":cp["step"],
                 "checkpoint_id":cp["checkpoint_id"],"context_id":digest(contract),
                 "contract":contract,**native_resources(qc)}
            save_json(out,row)
        rows.append(row)
    save_json(root/"native_resources/ecr-line/summary.json",rows)
    return len(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",default="results/quick")
    print("Synthetic target resource rows:",run_native(p.parse_args().output))


if __name__=="__main__":main()
