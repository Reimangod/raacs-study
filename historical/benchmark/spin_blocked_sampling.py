"""Qualified-per-draw tensor layout adapter, never a new physical ansatz."""
from dataclasses import asdict
import hashlib
import json
from time import perf_counter
import numpy as np
from .ansatz import circuit_bytes,measurement_equivalent_circuit,resources
from .common import digest
from .circuit_sampling_runtime import CircuitCountSampler
from .mps_layout_qualification import permute_sampling,logical_id
from .mps_sampling import construct_mps,sample_mps
from .study.core import rss_peak


class BlockedApproximationUnqualified(RuntimeError):
    pass


class SpinBlockedCountSampler(CircuitCountSampler):
    minimum_retained_norm_squared=1-1e-6

    def draw(self,preparation,shots,seed,n_parameters=0,n_operators=0):
        if self.spec.backend!='mps' or preparation.num_clbits or shots<1:
            raise ValueError('Spin-blocked MPS, static circuit and positive shots only')
        started=perf_counter();sampling,removed=measurement_equivalent_circuit(preparation)
        raw=circuit_bytes(sampling);physical_sha=hashlib.sha256(raw).hexdigest()
        blocked,mapping=permute_sampling(sampling)
        key=digest([physical_sha,asdict(self.spec),mapping,self.minimum_retained_norm_squared])
        hit=key in self.cache
        if hit:
            prepared,build=self.cache.pop(key);self.cache[key]=(prepared,build);build_seconds=0.
        else:
            tick=perf_counter();prepared,build=construct_mps(blocked,self.spec.max_bond,self.spec.cutoff,
                self.spec.construction_seconds_cap,self.spec.RSS_cap_bytes)
            build_seconds=perf_counter()-tick;self.MPS_builds+=1
            if build['retained_norm_squared']<self.minimum_retained_norm_squared:
                # No sampling has started. Carry all actually spent classical
                # build evidence in the preserved exception, not a fake count.
                raise BlockedApproximationUnqualified('resource cap: MPS approximation unqualified BEFORE_SAMPLING '+json.dumps({
                    'build':build,'construction_seconds_this_call':build_seconds,'confirmed_new_model_shots':0,
                    'logical_sampling_sha256':physical_sha,'logical_to_simulator_qubit':mapping}))
            if self.spec.cache_entries:
                self.cache[key]=(prepared,build)
                while len(self.cache)>self.spec.cache_entries:self.cache.popitem(last=False)
        ids,counts,sampled=sample_mps(prepared,shots,seed,self.spec.sampling_seconds_cap,self.spec.RSS_cap_bytes)
        logical=np.array([logical_id(x,mapping) for x in ids],dtype=np.uint64);order=np.argsort(logical)
        original_ids=ids.copy();original_counts=counts.copy();ids=logical[order];counts=counts[order]
        self.calls+=1;self.raw_model_shots+=int(counts.sum())
        meta={'sampling_circuit_sha256':physical_sha,'preparation_sampling_distinct':True,
            'removed_terminal_diagonal_gates':removed,'prepared_simulator_cache_hit':hit,
            'construction_seconds_this_call':build_seconds,'original_build_metadata':build,'sample_metadata':sampled,
            'sampling_resources':resources(sampling,n_parameters,n_operators),'preparation_resources':resources(preparation,n_parameters,n_operators),
            'call_wall_seconds':perf_counter()-started,'process_RSS_hwm_bytes':rss_peak(),'seed':str(seed),
            'requested_model_shots':shots,'actual_raw_model_shots':int(counts.sum()),'hardware_shots':0,'FCI_used':False,
            'full_statevector_exposed_to_algorithm':False,'layout_engine':'spin-blocked-simulator-only-v1',
            'minimum_required_retained_norm_squared':self.minimum_retained_norm_squared,
            'layout_evidence':{'logical_to_simulator_qubit':mapping,'backend_ids':original_ids.tolist(),
                               'backend_counts':original_counts.tolist(),'physical_gate_count_unchanged':True},
            'norm_is_not_global_error_certificate':True}
        return ids,counts,meta
