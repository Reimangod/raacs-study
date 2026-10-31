import sys
import pytest


@pytest.mark.parametrize("phase,module", [("scaling", "benchmark.molecular_mps_scaling"), ("early_stopping", "benchmark.actual_growth_cli")])
@pytest.mark.parametrize("eq", [False, True])
def test_required_phase_commands_route_without_starting_duplicate_work(monkeypatch, phase, module, eq):
    import importlib
    from benchmark.run import main
    destination = importlib.import_module(module); seen = []
    monkeypatch.setattr(destination, "main", lambda: seen.append(list(sys.argv)))
    original = ["benchmark.run", f"--phase={phase}", "--mode", "standard"] if eq else ["benchmark.run", "--phase", phase, "--mode", "standard"]
    monkeypatch.setattr(sys, "argv", original)
    main()
    assert seen == [["benchmark.run", "--mode", "standard"]]
    assert sys.argv is original
