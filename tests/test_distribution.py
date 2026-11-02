"""The curated distribution retains results and the registered source identity."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_local_agent_instructions_or_development_reports():
    for name in ('AGENTS.md', 'Agent.md', 'agent.md', 'CLAUDE.md',
                 'verification', 'configs/exploratory'):
        assert not (ROOT / name).exists(), name
    assert not list((ROOT / 'reference').rglob('test_*.py'))


def test_policy_registry_points_to_the_curated_source():
    registry = json.loads((ROOT / 'configs/policy-registry.json').read_text())
    assert registry['schema'] == 'policy-registry-2'
    assert registry['display_alias'] == {'B4': 'RAACS'}
    filename, name = registry['canonical_implementation'].split(':')
    assert name == 'SpinBeamEngine' and (ROOT / filename).is_file()
    for filename in registry['cohort_protocols']:
        assert (ROOT / filename).is_file()
    assert (ROOT / registry['source_index']).is_file()


def test_reference_sources_are_unchanged_and_hash_bound():
    index = json.loads((ROOT / 'data/source-files.json').read_text())
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    hashes = {entry['path']: entry['sha256'] for entry in manifest['files']}
    assert manifest['archive_release']['tag'] == 'data-v0.1.0'
    assert manifest['archive_release']['commit'] == 'e2fd71c77757ca0de2aecd5fc9f1664aba62e38a'
    assert len(index['files']) == 61
    for row in index['files']:
        path = ROOT / row['destination']
        assert path.is_relative_to(ROOT / 'reference')
        assert path.stat().st_size == row['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'] == hashes[row['destination']]
    assert 'Apache License' in (ROOT / 'reference/vendor/ceo/LICENSE').read_text()


def test_reference_static_import_closure_is_retained():
    index = json.loads((ROOT / 'data/source-files.json').read_text())
    entries = {entry['module']: entry for entry in index['files']}
    assert len(entries) == len(index['files'])
    reached = set()
    queue = index['import_roots'].copy()
    while queue:
        name = queue.pop()
        if name in reached:
            continue
        assert name in entries, name
        reached.add(name)
        row = entries[name]
        queue.extend(row['local_imports'])
        path = ROOT / row['destination']
        ast.parse(path.read_text())
    assert reached == set(entries)


def test_every_original_comparison_table_remains_hash_bound():
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    paths = {entry['path'] for entry in manifest['files']}
    for path in (ROOT / 'data/original-tables').glob('*.csv'):
        assert path.relative_to(ROOT).as_posix() in paths
    for filename in ('data/catalog.csv', 'data/original-rows.jsonl', 'data/cases.json',
                     'data/missing-sources.csv', 'data/source-population.csv',
                     'data/archive-manifest.json', 'data/source-inventory.jsonl.gz'):
        assert filename in paths
