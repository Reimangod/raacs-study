"""Persistent isolated exporter; uses unchanged, qualified native SDK export().

Per-member raw SDK files exist only in declared transaction scratch. Their exact
bytes are compressed losslessly before the temporary transaction is removed.
No old artifacts or active-runtime packages are changed.
"""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from .common import file_digest, save_binary, save_json
from .sqdrift_sdk_export import export


def serve(base_path, root):
    # Charge cold interpreter/import startup once per case, not to every seed.
    import qiskit
    import qiskit_fermions
    if qiskit.__version__ != "2.5.0":
        raise RuntimeError("Native SDK requires its isolated pinned Qiskit2.5 overlay")
    base = json.loads(Path(base_path).read_text())
    root = Path(root).resolve(); root.mkdir(exist_ok=True, parents=True)
    print(json.dumps({"ready": True, "qiskit": qiskit.__version__}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("stop"):
            break
        target = (root / request["relative_directory"]).resolve()
        if root not in target.parents:
            raise RuntimeError("Member output must stay within registered case directory")
        target.mkdir(exist_ok=True, parents=True)
        started = perf_counter()
        try:
            args = dict(base, time=request["time"], num_terms=request["num_terms"], seed=request["seed"])
            with tempfile.TemporaryDirectory(prefix="native-transaction-", dir=root) as scratch:
                export(args, scratch)
                for filename in ("sdk-circuit.qpy", "sdk-result.json"):
                    raw = (Path(scratch) / filename).read_bytes()
                    save_binary(target / (filename+".gz"), gzip.compress(raw, mtime=0))
            event = {"status": "SDK_MEMBER_EXPORTED", "request": request,
                     "sdk_circuit_gzip_sha256": file_digest(target / "sdk-circuit.qpy.gz"),
                     "sdk_result_gzip_sha256": file_digest(target / "sdk-result.json.gz"),
                     "SDK_wall_seconds": perf_counter()-started}
        except Exception as exc:
            import traceback
            event = {"status": "SDK_EXPORT_FAILED_PRESERVED", "request": request,
                     "error": repr(exc), "traceback": traceback.format_exc(), "SDK_wall_seconds": perf_counter()-started}
        save_json(target / "sdk-export-event.json", event)
        print(json.dumps(event), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--base", required=True); p.add_argument("--root", required=True)
    args = p.parse_args(); serve(args.base, args.root)
