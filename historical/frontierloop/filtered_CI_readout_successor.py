from pathlib import Path
from . import filtered_SqDRIFT_partial_readout as parent
from .contiguous_CI_audit import audit
from .io import load,save,sha256


def run():
    old=load(parent.LOCAL/'partial-CI-terminal-v1.json')
    assert old['completed_endpoints']==0 and old['actual_new_CI_roots']==0
    assert old['failed_CI_attempts']==18
    save(parent.LOCAL/'contiguous-RDM-successor-v2.json',dict(
        predecessor_SHA256=sha256((parent.LOCAL/'partial-CI-terminal-v1.json').read_bytes()),
        prior_CPU_seconds=old['CPU_seconds'],frozen_reader_SHA256=sha256((parent.LOCAL/'partial-CI-reader-v1-frozen.py').read_bytes()),
        audit_code_SHA256=sha256(Path(__import__('frontierloop.contiguous_CI_audit',fromlist=['audit']).__file__).read_bytes()),
        change='CONTIGUOUS_NUMPY_COPY_BEFORE_PYSCF_RAW_POINTER_RDM_ONLY',
        physics_bases_seeds_tolerances_unchanged=True,new_model_samples=0))
    parent.run(version='v2',audit_function=audit)


if __name__=='__main__':run()
