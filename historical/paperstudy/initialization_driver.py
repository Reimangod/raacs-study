"""Serial bounded preprocessing; not a replacement for the full study queue."""
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from .io import ROOT, save_json


def main():
    stages = []
    started = perf_counter()
    for case in ("N2-32-R1p2-P2", "N2-32-R2p4-P2", "FE2S2-40-P2"):
        command = ["rtk", "proxy", sys.executable, "-m", "paperstudy.author_compressed", "--case", case, "--fresh", "--seconds", "1200"]
        log_path = ROOT/"qualification"/(case+"-fresh-driver.log")
        record = {"case_id": case, "command": command, "status": "RUNNING", "driver_pid": os.getpid()}
        stages.append(record)
        save_json(ROOT/"initialization-driver.json", {"stage": "S1_C_LANE_PREPROCESSING", "stages": stages,
                                                     "whole_experiment_complete": False})
        with log_path.open("a") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        record["returncode"] = result.returncode
        record["status"] = "PROCESS_RETURNED" if result.returncode == 0 else "FAILED_PRESERVED"
        record["output"] = str(log_path.relative_to(ROOT))
        save_json(ROOT/"initialization-driver.json", {"stage": "S1_C_LANE_PREPROCESSING", "stages": stages,
                    "elapsed_seconds": perf_counter()-started, "whole_experiment_complete": False})
        print(case, record["status"], flush=True)
    print("PREPROCESSING_SEQUENCE_RETURNED; MAIN_STUDY_NOT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
