from benchmark.full_objective_audit_v2 import correct_items


def test_intro_targets_are_not_classified_as_artifact_preservation():
    rows=[dict(line='1',kind='heading',group='0',assessment='EXECUTED_VERIFIED',evidence='original'),
          dict(line='2',kind='list_item',group='0',assessment='EXECUTED_VERIFIED',evidence='original'),
          dict(line='4',kind='list_item',group='0',assessment='EXECUTED_VERIFIED',evidence='original')]
    text='# Study\n- fewer gates\n# 0. Source\n- inventory'
    judgments=[dict(group='12target',status='EVALUATED_NOT_UNIVERSALLY_ESTABLISHED',evidence='scoped-results')]
    out=correct_items(rows,text,judgments)
    assert out[0]['assessment']=='CONTEXT_TITLE_NOT_A_SCIENTIFIC_RESULT'
    assert out[1]['assessment']=='EVALUATED_NOT_UNIVERSALLY_ESTABLISHED'
    assert out[2]==rows[2]
    assert rows[1]['assessment']=='EXECUTED_VERIFIED'
