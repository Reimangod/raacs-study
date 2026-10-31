from unittest.mock import patch
import pytest
from benchmark.portable_report_entry import run,OBJECTIVE_SHA256


def test_portable_goal_rejects_even_a_normalizing_newline_change(tmp_path):
    source=tmp_path/'goal.md';source.write_text('Not the complete frozen objective\n')
    with pytest.raises(ValueError,match='frozen full owner objective'):
        run(source,tmp_path/'report')
    assert not (tmp_path/'report').exists()


def test_adapter_changes_only_goal_location_and_restores_provider(tmp_path):
    from benchmark import current_evidence
    old=current_evidence.OBJECTIVE;path=tmp_path/'objective.md'
    with patch.object(current_evidence,'sha',return_value=OBJECTIVE_SHA256),patch.object(current_evidence,'build') as builder:
        builder.side_effect=lambda output,mode:{'goal':current_evidence.OBJECTIVE,'mode':mode}
        answer=run(path,tmp_path/'new','quick')
        assert answer=={'goal':path.resolve(),'mode':'quick'}
        assert current_evidence.OBJECTIVE==old
