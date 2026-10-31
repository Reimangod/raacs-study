from benchmark.common import resolve_artifact


def test_relocated_artifacts_do_not_use_old_directory(tmp_path):
    p=resolve_artifact(tmp_path,"/old/project/results/distributions/H2_R0.7/hf/step-0000.npz")
    assert p==tmp_path/"distributions/H2_R0.7/hf/step-0000.npz"
    q=resolve_artifact(tmp_path,"/old/project/results/checkpoints/H2_R0.7/hf/preparation-0000.qpy")
    assert q==tmp_path/"checkpoints/H2_R0.7/hf/preparation-0000.qpy"
