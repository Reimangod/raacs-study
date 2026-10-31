from benchmark.config import Case
from benchmark.chemistry import build_reference
from benchmark.lucj import run_lucj


def test_random_lucj_does_not_require_ccsd(tmp_path,monkeypatch):
    import benchmark.lucj as lucj
    def forbidden(*a,**k):raise AssertionError("Random control must not call CCSD")
    monkeypatch.setattr(lucj,"ccsd_amplitudes",forbidden)
    case=Case("H2",.7);r=build_reference(case,tmp_path)
    status=run_lucj(case,r,"lucj_random_101",tmp_path)
    assert status["status"]=="FIXED_RANDOM"
    assert not (r["directory"]/"ccsd.json").exists()
