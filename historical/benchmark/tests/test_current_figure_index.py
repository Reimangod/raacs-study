import pytest
from benchmark.current_figure_index import CONTRACTS, validate_contracts


def test_all18_contracts_have_sources_and_semantic_boundaries():
    validate_contracts(CONTRACTS)
    assert len(CONTRACTS)==18
    for row in CONTRACTS:
        assert row[3].endswith('.png') and row[5]
        assert len(row[4])>100


def test_missing_or_duplicate_requested_relationship_fails():
    with pytest.raises(ValueError):validate_contracts(CONTRACTS[:-1])
    with pytest.raises(ValueError):validate_contracts(CONTRACTS[:-1]+[CONTRACTS[0]])


def test_semantics_do_not_substitute_old_incomplete_forms():
    records={r[0]:r for r in CONTRACTS}
    assert 'allK' in records[4][2]
    assert 'LOMO-cluster' in records[10][2]
    assert 'energy_pairs' in records[14][2]
    assert 'report-presentation-v2' in records[18][2]
    assert 'not historicalCEO' in records[18][4]


def test_actual_repository_contract_inputs_exist():
    from benchmark.current_figure_index import ROOT
    for row in CONTRACTS:
        assert (ROOT/row[3]).is_file(), row[3]
        for path in row[5]:assert (ROOT/path).is_file(),path
