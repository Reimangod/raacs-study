"""N24 known hard-core pair-Givens resource qualification; NO molecular energy.

Not a new ansatz, DOCI-QSCI reproduction, or performance/accuracy experiment.
The angle schedule below is a synthetic compiler probe, not CCSD initialization.
"""
from io import BytesIO
from pathlib import Path
from time import process_time,perf_counter
import importlib.metadata
import xml.etree.ElementTree as ET
import numpy as np
from qiskit import QuantumCircuit,qpy
from qiskit.circuit.library import XXPlusYYGate
from .paths import ROOT
from .io import load,save,write,sha256,space
from .slater_prefix_preflight import compile_views

LOCAL=ROOT/'runs/N24-PAIRED-RESOURCE-PREFLIGHT-v1'


def schedule(n,k):
    if not 0<k<n:raise ValueError('Nontrivial paired fixed-particle sector required')
    v=n-k
    if k>=v:return [[((t+j)%k,k+j) for j in range(v)] for t in range(k)]
    return [[(i,k+(t+i)%v) for i in range(k)] for t in range(v)]


def circuit(n,k,copy_spin_register=False):
    q=QuantumCircuit(2*n if copy_spin_register else n)
    for i in range(k):q.x(i)
    e=0
    for layer in schedule(n,k):
        for i,a in layer:
            # Positive target amplitude is independently tested against JW pair K.
            q.append(XXPlusYYGate(2*(.173+.013*(e%7)),np.pi/2),[i,a]);e+=1
    if copy_spin_register:
        for i in range(n):q.cx(i,n+i)
    return q


def prepare():
    space()
    if LOCAL.exists():raise RuntimeError('Existing N24; preserve, do not overwrite/restart')
    test=ROOT/'qualification-N24-pair-preexecution-v2-tests.xml'
    suites=list(ET.parse(test).getroot().iter('testsuite'))
    if sum(int(s.get('tests','0')) for s in suites)!=3 or any(int(s.get('failures','0')) or int(s.get('errors','0')) for s in suites):raise RuntimeError('All3preexecution qualification tests required')
    sources=load(ROOT/'runs/N23-BACKBONE-UNIFORM-SOURCES-v1/protocol.json')['sources']
    if len(sources)!=4 or any(s['nelec'][0]!=s['nelec'][1] for s in sources):raise RuntimeError('All4same balanced physical cases required')
    LOCAL.mkdir()
    save(LOCAL/'protocol.json',dict(version='N24_KNOWN_PAIR_GIVENS_RESOURCES_ONLY_V1',sources=sources,
        qualification_SHA256=sha256(test.read_bytes()),code_SHA256=sha256(Path(__file__).read_bytes()),
        compiler_helper_SHA256=sha256((ROOT/'frontierloop/slater_prefix_preflight.py').read_bytes()),
        packages={n:importlib.metadata.version(n) for n in ('numpy','qiskit','ffsim')},
        angle_rule='SYNTHETIC_0.173_PLUS_0.013_TIMES_EDGE_INDEX_MOD7_NOT_MOLECULAR_COEFFICIENTS',
        scheduling='COMPLETE_OCCUPIED_VIRTUAL_BIPARTITE_MATCHINGS_FIXED_ORDER',
        compiler_levels=[1,3],seed_transpiler=8137,approximation_degree=1.,
        views=['SPATIAL_PAIR_ENCODING','FULL_SPIN_REGISTER_CNOT_COPY'],
        declared_parameter_rule='ONE_REGISTERED_REAL_VARIABLE_PER_PAIR_EDGE_NOT_MINIMUM_JACOBIAN_RANK',
        quantum_sampling_classical_readout_energies_and_optimizers_authorized=False,
        new_model_shots=0,new_QPU_shots=0,new_CI_roots=0,GPU_allocations=0,
        known_method_not_novelty=True,goal_complete=False,max_wall_seconds=180))
    print('N24_RESOURCES_ONLY_FROZEN_ALL4_NO_MOLECULAR_OUTCOMES',flush=True)


def run():
    space();p=load(LOCAL/'protocol.json')
    if (LOCAL/'started.json').exists():raise RuntimeError('Existing intent; reconcile, never blindly restart')
    if sha256(Path(__file__).read_bytes())!=p['code_SHA256'] or sha256((ROOT/'frontierloop/slater_prefix_preflight.py').read_bytes())!=p['compiler_helper_SHA256']:raise RuntimeError('Frozen compiler/implementation differs')
    if {n:importlib.metadata.version(n) for n in p['packages']}!=p['packages']:raise RuntimeError('Pinned runtime differs')
    save(LOCAL/'started.json',dict(protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    cpu,wall=process_time(),perf_counter();rows=[]
    for s in p['sources']:
        n=s['norb'];k=s['nelec'][0]
        for full in (False,True):
            view=p['views'][int(full)];qc=circuit(n,k,full)
            for level in p['compiler_levels']:
                if perf_counter()-wall>p['max_wall_seconds']:raise TimeoutError('Resource-only global wall cap')
                prep,samp,res=compile_views(qc,level)
                files={}
                for role,q in [('preparation',prep),('sampling',samp)]:
                    stream=BytesIO();qpy.dump(q,stream);blob=stream.getvalue()
                    path=LOCAL/s['case']/view/f'opt{level}-{role}.qpy';write(path,blob)
                    actual=qpy.load(BytesIO(blob))[0]
                    if actual.count_ops()!=q.count_ops() or actual.depth()!=q.depth():raise RuntimeError('QPY roundtrip failed')
                    files[role]=dict(path=str(path),SHA256=sha256(blob))
                rows.append(dict(case=s['case'],problem_id=s['problem_id'],norb=n,k=k,view=view,
                    n_qubits=qc.num_qubits,declared_parameters=k*(n-k),pair_gate_layers=len(schedule(n,k)),
                    resources=res,circuits=files,status='SYNTHETIC_RESOURCE_PROBE_ONLY_NO_PRECISION_LABEL'))
    result=dict(status='N24_ALL4_CASES16_COMPILED_POINTS_COMPLETE',rows=rows,
        CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
        protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),
        new_model_shots=0,new_QPU_shots=0,new_CI_roots=0,GPU_allocations=0,
        molecular_initialization_verified=False,molecular_accuracy_verified=False,goal_complete=False)
    save(LOCAL/'terminal.json',result)
    lines=['# N24：既知の電子対回路の資源予備検査','',
        '角度は事前固定の合成値。分子の良い波動関数を得た実験ではない。電子対encodingとfull-spinコピーを別々に報告する。',
        'pair encodingのsamplingをalpha/betaへ展開するのは古典処理であり、展開後CI状態の量子準備とは違う。',
        '', '| 条件 | 表現 | qubit | opt | CNOT | CX Depth | Total Depth | 宣言Parameters |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        z=r['resources'];lines.append(f"| {r['case']} | {r['view']} | {r['n_qubits']} | {z['optimization_level']} | {z['CNOT']} | {z['CNOT_Depth']} | {z['Total_Depth']} | {r['declared_parameters']} |")
    lines+=['',f"CPU {result['CPU_seconds']:.6f}秒。新Shot・分子energy・CI・optimizer・GPU割当は0。",
        '既知のpUCCD型pair-Givens資源検査であり、DOCI-QSCIの原著spinless LUCJやAFQMCを再現したものではない。',
        '実際のCCSD初期化の符号/規格化、軌道依存、broken-pair相関不足、低Shotでの精度と古典対照からの利益は未検証。',
        '元LUCJと同じcompiler規則でも、精度が異なるため資源上の優越とは判定しない。回路を小さく作れることだけでは研究goalを満たさない。']
    write(ROOT/'analysis/N24_PAIRED_RESOURCE_RESULTS_JA.md',('\n'.join(lines)+'\n').encode())
    print(result['status'],result['CPU_seconds'],flush=True)


if __name__=='__main__':
    import sys
    {'prepare':prepare,'run':run}[sys.argv[1]]()
