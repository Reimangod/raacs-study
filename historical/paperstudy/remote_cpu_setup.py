"""Additive isolated CPU runtime. Never installs into an old environment."""
import json
from pathlib import Path
import subprocess
import sys


def main(target):
    directory = Path(target)
    marker = directory/"runtime.json"
    if marker.exists():
        print(marker.read_text())
        return
    directory.mkdir(parents=True, exist_ok=True)
    packages = ["numpy==2.4.6", "scipy==1.17.1", "ffsim==0.0.84", "pyscf==2.14.0", "qiskit==2.6.0", "jax==0.10.2"]
    command = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--target", str(directory/"packages"), *packages]
    subprocess.run(command, check=True)
    import os
    environment = dict(os.environ, PYTHONPATH=str(directory/"packages"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    result = subprocess.check_output([sys.executable, "-c", "import ffsim,pyscf,qiskit; import importlib.metadata as m,json;print(json.dumps({p:m.version(p) for p in ['numpy','scipy','ffsim','pyscf','qiskit','jax']}))"], env=environment, text=True)
    marker.write_text(json.dumps({"runtime": json.loads(result), "command": command,
                     "legacy_environment_modified": False}, indent=2)+"\n")
    print(marker.read_text())


if __name__ == "__main__":
    main(sys.argv[1])
