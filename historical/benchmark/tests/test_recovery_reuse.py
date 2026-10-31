from benchmark.selected_recovery import reusable_final_rows


def test_recovery_iteration_mismatch_is_not_reused():
    data={"context":{"iterations":5,"primary_iteration":"last-not-best"},"rows":[{"source_raw_evaluation_id":"x"}]}
    assert reusable_final_rows(data)=={}
    data["context"]["iterations"]=3
    assert "x" in reusable_final_rows(data)
    data["context"]["primary_iteration"]="best"
    assert reusable_final_rows(data)=={}
