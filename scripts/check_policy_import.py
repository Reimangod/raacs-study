"""Check that archived policy imports resolve inside a specified fresh clone.

Run with the historical-policy dependencies available. No policy is executed.
"""
import argparse,sys,json,hashlib,inspect
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--repo",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
a=p.parse_args();repo=a.repo.resolve();sys.path.insert(0,str(repo/"historical"))
from frontierloop.spin_beam import SpinBeamEngine
import frontierloop,raacsnext,paperstudy,benchmark,adaptvqe
modules={x.__name__:str(Path(x.__file__).resolve())for x in(frontierloop,raacsnext,paperstudy,benchmark,adaptvqe)}
outside={k:v for k,v in modules.items()if not Path(v).is_relative_to(repo/"historical")}
result={"passed":not outside,"scope":"import only, using available historical dependencies; no source execution or molecular sampling",
        "policy_class":SpinBeamEngine.__name__,"source_sha256":hashlib.sha256(Path(inspect.getfile(SpinBeamEngine)).read_bytes()).hexdigest(),
        "module_files":modules,"outside_snapshot":outside,"new_model_samples":0}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
if outside:raise SystemExit(1)
