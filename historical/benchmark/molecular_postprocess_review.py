"""Read-only QPY/resource/CSV review; no MPS construction, transpilation or CI."""
import argparse
from collections import Counter
import csv
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return '' if value is None else str(value)


def check_csv(path, expected, keys):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream); actual = list(reader); fields = reader.fieldnames
    lookup = {}
    for row in expected:
        key = tuple(csv_value(row[k]) for k in keys)
        if key in lookup:
            raise ValueError('Duplicate expected CSV row')
        lookup[key] = row
    seen = set()
    for row in actual:
        key = tuple(row[k] for k in keys)
        if key in seen or key not in lookup:
            raise ValueError('Unexpected/duplicate CSV row')
        seen.add(key)
        if any(row[k] != csv_value(lookup[key].get(k)) for k in fields):
            raise ValueError('CSV value mismatch')
    if seen != set(lookup):
        raise ValueError('CSV dropped planned rows')
    return len(actual)


def check_compiled(qc, resource, nqubits):
    if qc.num_qubits != nqubits or qc.num_clbits != nqubits:
        raise ValueError('Qubit count mismatch')
    mapping = resource['measurement_physical_to_logical_clbit']
    pairs = sorted((int(a), int(b)) for a, b in mapping)
    if sorted(a for a, _ in pairs) != list(range(nqubits)) or sorted(b for _, b in pairs) != list(range(nqubits)):
        raise ValueError('Incomplete/ambiguous readout map')
    actual = sorted((qc.find_bit(g.qubits[0]).index, qc.find_bit(g.clbits[0]).index)
                    for g in qc.data if g.operation.name == 'measure')
    if actual != pairs:
        raise ValueError('Physical/logical readout mismatch')
    if dict(qc.count_ops()) != resource['gate_counts']:
        raise ValueError('Native gate histogram mismatch')
    if int(qc.count_ops().get('cx', 0)) != resource['native_2q_count']:
        raise ValueError('Native CX count mismatch')
    depth = qc.depth(filter_function=lambda g: g.operation.num_qubits == 2)
    if depth != resource['native_2q_depth']:
        raise ValueError('Native depth mismatch')
    if qc.remove_final_measurements(inplace=False).depth() != resource['native_total_gate_depth_without_measurement']:
        raise ValueError('Total native depth mismatch')
    if resource['native_2q_family'] != 'CX' or not resource['synthetic_not_calibrated_hardware']:
        raise ValueError('Wrong target/gate-family interpretation')
    for gate in qc.data:
        if gate.operation.name not in ('rz', 'sx', 'x', 'cx', 'measure', 'barrier'):
            raise ValueError('Non-native gate')
        if gate.operation.name == 'barrier':
            continue
        if gate.operation.num_qubits == 2 and gate.operation.name != 'cx':
            raise ValueError('Non-native 2Q gate')
        if resource['target'] == 'line_cx' and gate.operation.num_qubits == 2:
            a, b = (qc.find_bit(q).index for q in gate.qubits)
            if abs(a - b) != 1:
                raise ValueError('Line connectivity violation')


def review(root, output):
    from qiskit import qpy
    root, output = Path(root).resolve(), Path(output).resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError('Write outside the frozen producer tree')
    inputs = {}
    def read(path, expected=None):
        path = Path(path); raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError('Bound source changed: ' + str(path))
        inputs[str(path)] = digest
        return json.loads(raw)
    cfg = read(root / 'manifest.json')
    state = read(root / 'execution-status.json')
    if state['status'] != 'MOLECULAR_NATIVE_AND_BOND_ANALYSIS_EXECUTED_REPORTED':
        raise ValueError('All producer records and report required')
    for name, digest in cfg['implementation'].items():
        if sha(ROOT / 'benchmark' / name) != digest:
            raise ValueError('Producer numerical source changed')
    source = Path(cfg['source'])
    read(source / 'manifest.json', cfg['source_manifest_sha256'])
    validation = read(source / 'validation.json', cfg['source_validation_sha256'])
    layout = read(source / 'physical-layout-validation-v2.json', cfg['source_physical_layout_validation_sha256'])
    original = {(r['case'], r['method']): r for r in validation['rows']}
    planned = {(item['id'], method) for item in cfg['queue'] for method in cfg['methods']}
    report = root / 'report-v1'; report_manifest = read(report / 'manifest.json')
    if report_manifest['source_config_sha256'] != sha(root / 'manifest.json'):
        raise ValueError('Wrong report binding')
    for name, digest in report_manifest['files'].items():
        path = report / name
        if sha(path) != digest:
            raise ValueError('Changed report output')
        inputs[str(path)] = digest
    statuses = []; native = []; bonds = []; seen = set(); circuits = 0
    native_counts, bond_counts = Counter(), Counter()
    for path in sorted((root / 'cells').glob('*/result.json')):
        r = read(path); key = r['case'], r['method']
        if key not in planned or key in seen:
            raise ValueError('Wrong/duplicate planned cell')
        seen.add(key)
        old = original[key]
        if r['source_status'] != old['status']:
            raise ValueError('Source failure/terminal status relabelled')
        if 'source_result_sha256' in r and r['source_result_sha256'] != old['result_sha256']:
            raise ValueError('Wrong original result binding')
        if r['new_model_shots'] != 0 or r['hardware_shots'] != 0:
            raise ValueError('Unexpected new acquisition')
        statuses.append({k: r.get(k) for k in ('case','method','status','source_status','n_qubits','wall_seconds','peak_process_RSS_hwm')})
        if r['method'] == 'classical-SCI64' and (r.get('native') or r.get('MPS')):
            raise ValueError('Classical SCI falsely represented as a quantum circuit')
        for n in r.get('native', []):
            native_counts[n['status']] += 1
            native.append({'case':r['case'],'method':r['method'],'n_qubits':r['n_qubits'], **n})
            if n['status'] == 'COMPLETED':
                archive = path.parent / (n['target'] + '.qpy.z')
                if sha(archive) != n['compiled_qpy_sha256']:
                    raise ValueError('Compiled QPY changed')
                inputs[str(archive)] = n['compiled_qpy_sha256']
                qc, = qpy.load(BytesIO(zlib.decompress(archive.read_bytes())))
                check_compiled(qc, n, r['n_qubits']); circuits += 1
        mps = r.get('MPS', {})
        overlap = mps.get('normalized_overlap_squared')
        if overlap is not None and (not math.isfinite(overlap) or not -1e-8 <= overlap <= 1+1e-8):
            raise ValueError('Invalid saved overlap; not a recomputed error certificate')
        for b in mps.get('rows', []):
            bond_counts[b['status']] += 1
            norm = b.get('metadata', {}).get('retained_norm_squared')
            if b['status'] == 'COMPLETED' and (norm is None or not math.isfinite(norm) or not 0 <= norm <= 1+1e-8):
                raise ValueError('Invalid saved retained norm')
            bonds.append(dict(case=r['case'],method=r['method'],n_qubits=r['n_qubits'],bond=b['bond'],status=b['status'],
                retained_norm_squared=norm, normalized_overlap_squared_128_256=overlap,wall_seconds=b['wall_seconds']))
    if seen != planned or len(statuses) != state['terminal'] or state['terminal'] != state['planned']:
        raise ValueError('Incomplete planned denominator')
    check_csv(report / 'all-planned-status.csv', statuses, ('case', 'method'))
    check_csv(report / 'native-resources.csv', native, ('case', 'method', 'target'))
    check_csv(report / 'bond-convergence.csv', bonds, ('case', 'method', 'bond'))
    if any(sha(p) != digest for p, digest in inputs.items()):
        raise ValueError('Source drift during read-only review')
    out = dict(status='ALL_PLANNED_STATUS_CSV_NATIVE_QPY_COUNTS_DEPTHS_AND_READOUT_MAPS_REVIEWED',
        planned=len(planned), native_QPYs_rebuilt=circuits, native_rows=len(native), bond_rows=len(bonds),
        source_physical_layout_qualification=layout['status'],
        record_status_counts=dict(Counter(r['status'] for r in statuses)),
        native_status_counts=dict(native_counts), bond_status_counts=dict(bond_counts),
        input_sha256=inputs, review_code_sha256=sha(__file__),new_model_shots=0,new_MPS_builds=0,
        new_CI_solves=0, native_distribution_equivalence_recomputed=False,
        MPS_norms_and_overlaps_recomputed=False,whole_goal_complete=False)
    serialized = json.dumps(out, sort_keys=True, indent=2, allow_nan=False) + '\n'
    output.mkdir(parents=True, exist_ok=True); target = output / 'validation.json'
    if target.exists() and target.read_text() != serialized:
        raise ValueError('Do not overwrite a different review')
    if not target.exists(): target.write_text(serialized)
    print(json.dumps({k:v for k,v in out.items() if k != 'input_sha256'}, indent=2))
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); review(a.root, a.output)
