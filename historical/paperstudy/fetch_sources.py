"""Fetch only the reviewed author module, not a large repository/environment."""
import json
from pathlib import Path
import urllib.request

from .input_adapter import C1_COMMIT, BASE
from .io import ROOT, atomic_bytes, save_json, sha256


def fetch():
    relative = "src/lucj/operator_task/lucj_compressed_t2_task_ffsim/compressed_t2.py"
    path = ROOT / "vendor/C1" / relative
    if not path.exists():
        with urllib.request.urlopen(BASE+relative, timeout=30) as response:
            data = response.read(2**20)
        if b"Copyright IBM 2025" not in data or b"Apache License" not in data:
            raise RuntimeError("Unexpected reviewed author source/license")
        atomic_bytes(path, data)
    data = path.read_bytes()
    report = {"repository": "kevinsung/lucj", "commit": C1_COMMIT, "path": relative,
              "file_sha256": sha256(data), "unmodified_author_source": True,
              "source_comment_JAX_x64_default": "Not enabled in module; runtime precision must be recorded",
              "contains_author_copyright_notice": True}
    save_json(ROOT / "vendor/C1/source.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    fetch()
