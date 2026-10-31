"""B4: B3 candidate paths scored in the registered exact singlet space.

Same depth/beam/shot limits; exact-basis caching reduces redundant CPU work.
Proposals still use only physical integrals and paid observed configurations.
"""
from raacsnext.policy import action_key
from raacsnext.sampler import balanced_counts
from .io import digest
from .beam import BeamEngine
from .singlet_cache import SingletEnergyCache


class SpinBeamEngine(BeamEngine):
    def __init__(self,meta,h,g,journal,directory,budget,CPU_cap):
        super().__init__(meta,h,g,journal,directory,budget,CPU_cap)
        self.spin_cache=SingletEnergyCache(meta,h,g)

    def energy(self,bank,r=8):
        self.boundary()
        order=sorted(bank,key=action_key)
        counts,_=balanced_counts([x['observed'] for x in order],int(digest(
            ['balanced-pilot',self.journal.run_id,
             [(m['id'],sorted(m['observed'].items())) for m in order]])[:16],16))
        before=self.spin_cache.calls,self.spin_cache.hits,self.spin_cache.matvecs
        value=self.spin_cache.energy(counts,r)
        self.ci_calls+=self.spin_cache.calls-before[0]
        self.ci_cache_hits+=self.spin_cache.hits-before[1]
        self.ci_matvecs+=self.spin_cache.matvecs-before[2]
        return value

    def run(self):
        result=super().run()
        result.update(exact_singlet_policy_score=True,
            spin_score_distinct_basis_solves=self.spin_cache.calls,
            spin_score_same_basis_cache_hits=self.spin_cache.hits,
            spin_score_matvec_calls=self.spin_cache.matvecs,
            spin_score_max_selected_dimension=max(
                (x['spin_basis']['determinant_dimension'] for x in self.spin_cache.cache.values()),default=0),
            spin_score_min_singlet_dimension=min(
                (x['spin_basis']['singlet_dimension'] for x in self.spin_cache.cache.values()),default=0),
            cached_energy_scope='fixed Hamiltonian/solver/exact selected basis; no reused quantum evaluation')
        return result
