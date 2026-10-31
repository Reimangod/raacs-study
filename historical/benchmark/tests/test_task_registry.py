import pytest
from benchmark.task_registry import Registry


def test_shared_claim_prevents_cpu_gpu_duplicate(tmp_path):
    path=tmp_path/"tasks.db";a=Registry(path);b=Registry(path)
    task=a.claim({"case":"N2_R0.9","protocol":"test"},"A100-1")
    with pytest.raises(RuntimeError):b.claim({"protocol":"test","case":"N2_R0.9"},"M4")
    with pytest.raises(RuntimeError):b.update(task,"M4","COMPLETED",{})
    a.update(task,"A100-1","SUBMITTED",{"job":1})
    with pytest.raises(RuntimeError):b.claim({"case":"N2_R0.9","protocol":"test"},"A100-2")
    assert b.rows()[0][2]=="SUBMITTED"
