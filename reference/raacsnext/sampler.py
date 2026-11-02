"""Explicit new-ledger sampling adapter; old study writers are never called."""
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from time import perf_counter, process_time
import json

import numpy as np
from qiskit import QuantumCircuit, qpy
from benchmark.local_sampler_primitives import LocalQE
from paperstudy.budget import BudgetLedger
from benchmark.circuit_sampling_runtime import SamplingSpec
from paperstudy.sparse_raacs import ExactSparseSampler, traced_append
from .io import save, load, write_bytes, array_bytes, digest, sha256, space_check, new_path


def circuit_from_actions(meta, actions):
    q = QuantumCircuit(meta['n_qubits'])
    for wire in range(meta['n_qubits']):
        if (meta['hf_id'] >> wire) & 1:
            q.x(wire)
    for action in actions:
        op = LocalQE(tuple(action['source']), tuple(action['target']), meta['n_qubits'])
        q = op.append(q, action['theta'])
    q.metadata = {**(q.metadata or {}), 'paper_exact_sparse_actions':actions}
    return q


def merge_counts(observations):
    result = {}
    for observed in observations:
        for x,n in observed.items():
            if (not isinstance(x,(int,np.integer)) or isinstance(x,(bool,np.bool_)) or x < 0
                or not isinstance(n,(int,np.integer)) or isinstance(n,(bool,np.bool_)) or n <= 0):
                raise ValueError('Only actual positive integer observations may be pooled')
            result[int(x)] = result.get(int(x),0)+int(n)
    return result


def balanced_counts(observations, seed):
    """Without-replacement subsets; never create fractional or synthetic shots."""
    if not observations:
        raise ValueError('Empty circuit bank')
    for observed in observations:
        if not observed:raise ValueError('Empty member observations')
        merge_counts([observed])
    totals = [sum(x.values()) for x in observations]
    size = min(totals)
    rng = np.random.default_rng(seed)
    subsets = []
    for observed,total in zip(observations,totals):
        ids = sorted(observed)
        values = np.array([observed[x] for x in ids],np.int64)
        selected = values if total == size else rng.multivariate_hypergeometric(values,size)
        subsets.append({x:int(n) for x,n in zip(ids,selected) if n})
    return merge_counts(subsets), {'source_observations':sum(totals),
        'used_existing_observations':size*len(observations), 'per_member':size,
        'new_model_shots':0, 'sampling':'without-replacement hypergeometric'}


class Journal:
    def __init__(self, provider, directory, ledger_path, run_id, cap):
        self.provider = provider
        self.directory = Path(directory)
        self.run_id = run_id
        self.ledger = BudgetLedger(new_path(ledger_path))
        self.ledger.register(run_id,int(cap))
        self.new_draws = self.replayed_draws = 0
        self.MPS_builds = self.exact_small_builds = 0
        self.records = []

    def draw(self, preparation, shots, seed, n_parameters=0, n_operators=0, role=None):
        trace = (preparation.metadata or {}).get('paper_exact_sparse_actions',[])
        if len(trace) != n_operators:
            raise ValueError('Actual physical action trace required')
        cid = digest(['physical-QE-actions',self.provider.hf_id,preparation.num_qubits,trace])
        request = {'circuit_id':cid,'actions':trace,'hf_id':self.provider.hf_id,
            'n_qubits':preparation.num_qubits,'n_parameters':n_parameters,'shots':int(shots),
            'seed':str(seed),'role':role or 'native-seed-bound-draw'}
        request = json.loads(json.dumps(request))
        event = digest(request)
        folder = self.directory/'draws'/event
        evidence = folder/'draw.json'
        if not evidence.exists():space_check(1024**2)
        state = self.ledger.reserve(self.run_id,event,request,int(shots))
        if evidence.exists():
            record = load(evidence)
            if record['request'] != request or sha256((folder/'counts.npz').read_bytes()) != record['counts_SHA256']:
                raise RuntimeError('Saved draw identity or exact bytes changed')
            self.ledger.complete(self.run_id,event,record['actual_shots'],record['counts_SHA256'])
            with np.load(folder/'counts.npz',allow_pickle=False) as a:
                ids,counts = a['ids'].copy(),a['counts'].copy()
            self.replayed_draws += 1
            self.records.append(record)
            return ids,counts,record['metadata']
        if state != 'NEW_RESERVATION':
            raise RuntimeError('Unknown prior draw outcome; preserve reservation, do not remeasure')
        tick,cpu = perf_counter(),process_time()
        ids,counts,metadata = self.provider.draw(preparation,shots,seed,n_parameters,n_operators)
        blob = array_bytes(ids=ids,counts=counts)
        write_bytes(folder/'counts.npz',blob)
        circuit_path = self.directory/'circuits'/(cid+'.qpy')
        if not circuit_path.exists():
            stream = BytesIO(); qpy.dump(preparation,stream)
            write_bytes(circuit_path,stream.getvalue())
            save(circuit_path.with_suffix('.json'),{'circuit_id':cid,'actions':trace,
                'preparation_resources':metadata['preparation_resources'],
                'sampling_resources':metadata['sampling_resources']})
        record = {'event_id':event,'request':request,'actual_shots':int(counts.sum()),
            'counts_SHA256':sha256(blob),'metadata':metadata,'CPU_seconds':process_time()-cpu,
            'wall_seconds':perf_counter()-tick,'actual_QPU_shots':0}
        save(evidence,record)
        self.ledger.complete(self.run_id,event,record['actual_shots'],record['counts_SHA256'])
        self.new_draws += 1
        self.records.append(record)
        return ids,counts,metadata

    def measure(self, meta, actions, shots, role):
        seed = int(digest([self.run_id,role])[:16],16)
        q = circuit_from_actions(meta,actions)
        ids,counts,metadata = self.draw(q,int(shots),seed,len(actions),len(actions),role=role)
        observed = {int(x):int(n) for x,n in zip(ids,counts)}
        return {'id':digest(['physical-QE-actions',meta['hf_id'],meta['n_qubits'],actions]),
            'actions':actions,'observed':observed,'metadata':metadata,'circuit':q}


def provider(meta):
    return ExactSparseSampler(SamplingSpec(backend='exact_sparse_QE',cache_entries=2),meta['hf_id'])
