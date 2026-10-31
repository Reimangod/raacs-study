"""Streaming FCI-free rules: no trajectory length, future energies or labels.

Engineering kernel for PhaseC, not evidence of online molecular performance.
Thresholds below are registered design assumptions, not error certificates.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class ObservedFrame:
    problem_id: str
    trajectory_id: str
    step: int
    qsci_energy_Ha: float
    actual_dimension: int
    new_observed_valid_nonHF: int
    cumulative_pilot_raw_shots: int
    hf_raw_fraction: float
    cnot_count: int
    cnot_depth: int
    cumulative_classical_seconds: float
    current_pool_gradient_norm: float | None = None
    external_residual_norm_Ha: float | None = None
    pt2_proxy_Ha: float | None = None
    residual_scope: str = "NOT_EVALUATED"

    def __post_init__(self):
        if not self.problem_id or not self.trajectory_id:
            raise ValueError("Bind every observation to its physical problem and trajectory")
        if self.step < 0 or self.actual_dimension < 1 or self.new_observed_valid_nonHF < 0:
            raise ValueError("Invalid observed frame counts")
        if self.cnot_count < 0 or self.cnot_depth < 0 or self.cumulative_pilot_raw_shots < 0:
            raise ValueError("Invalid observed resource counts")
        if not 0 <= self.hf_raw_fraction <= 1 or not math.isfinite(self.qsci_energy_Ha):
            raise ValueError("Nonfinite energy or invalid HF rate")
        if not math.isfinite(self.cumulative_classical_seconds) or self.cumulative_classical_seconds < 0:
            raise ValueError("Invalid elapsed classical work")
        for value in (self.current_pool_gradient_norm, self.external_residual_norm_Ha, self.pt2_proxy_Ha):
            if value is not None and not math.isfinite(value):
                raise ValueError("Unknown meansNone, notNaN/infinity/zero")
        for value in (self.current_pool_gradient_norm, self.external_residual_norm_Ha):
            if value is not None and value < 0:
                raise ValueError("A norm cannot be negative")
        if (self.external_residual_norm_Ha is not None or self.pt2_proxy_Ha is not None) and self.residual_scope == "NOT_EVALUATED":
            raise ValueError("Residual/PT2 proxies require an explicit evaluation scope")


@dataclass(frozen=True)
class RuleSpec:
    name: str
    energy_plateau_Ha: float = 5e-4
    improvement_per_added_cnot_Ha: float = 1e-5
    pt2_proxy_threshold_Ha: float = 1e-4
    new_configurations_threshold: int = 1
    current_gradient_threshold: float = 1e-3
    patience: int = 2
    minimum_probes: int = 3


def registered_rules():
    return [RuleSpec(name) for name in ("gradient", "energy_plateau", "improvement_per_cnot", "discovery_plateau", "residual_pt2", "combined_cost")]


class OnlineStopRule:
    """A policy can only receive frames already observed, one at a time."""
    def __init__(self, spec):
        if spec.name not in {s.name for s in registered_rules()}:
            raise ValueError("Unregistered stopping rule")
        if spec.patience < 1 or spec.minimum_probes < 1 or any(not math.isfinite(x) or x < 0 for x in (
            spec.energy_plateau_Ha, spec.improvement_per_added_cnot_Ha, spec.pt2_proxy_threshold_Ha, spec.current_gradient_threshold)):
            raise ValueError("Invalid stopping threshold/patience")
        self.spec = spec; self.frames = []; self.consecutive = 0; self.decision = None

    def observe(self, frame):
        if self.decision is not None:
            raise RuntimeError("Stopped policy must not inspect a later frame")
        if self.frames:
            prev = self.frames[-1]
            if (prev.problem_id, prev.trajectory_id) != (frame.problem_id, frame.trajectory_id):
                raise ValueError("A stopping episode cannot mix different problems/trajectories")
            if frame.step <= prev.step or frame.cumulative_pilot_raw_shots < prev.cumulative_pilot_raw_shots:
                raise ValueError("Observation order / shared shot ledger regression")
            if frame.cumulative_classical_seconds < prev.cumulative_classical_seconds:
                raise ValueError("Cumulative cost cannot decrease")
        else:
            prev = None
        self.frames.append(frame); s = self.spec
        improvement = None if prev is None else prev.qsci_energy_Ha-frame.qsci_energy_Ha
        energy_flat = improvement is not None and improvement <= s.energy_plateau_Ha
        discovery_flat = frame.new_observed_valid_nonHF <= s.new_configurations_threshold
        # A partial external residual/PT2 is a proxy, not an accuracy bound.
        pt2_small = frame.pt2_proxy_Ha is not None and abs(frame.pt2_proxy_Ha) <= s.pt2_proxy_threshold_Ha
        added = None if prev is None else frame.cnot_count-prev.cnot_count
        gate_flat = improvement is not None and added is not None and added > 0 and improvement/added <= s.improvement_per_added_cnot_Ha
        gradient_small = frame.current_pool_gradient_norm is not None and frame.current_pool_gradient_norm <= s.current_gradient_threshold
        condition = {"gradient": gradient_small, "energy_plateau": energy_flat,
                     "improvement_per_cnot": gate_flat, "discovery_plateau": discovery_flat,
                     "residual_pt2": pt2_small and energy_flat,
                     "combined_cost": discovery_flat and energy_flat and (gate_flat or pt2_small)}[s.name]
        self.consecutive = self.consecutive+1 if condition else 0
        required = 1 if s.name == "gradient" else s.patience
        sufficient = s.name == "gradient" or len(self.frames) >= s.minimum_probes
        if sufficient and self.consecutive >= required:
            self.decision = {"stop": True, "step": frame.step, "rule": asdict(s), "observed_probes": len(self.frames),
                             "cumulative_pilot_raw_shots": frame.cumulative_pilot_raw_shots,
                             "reason": s.name, "FCI_used": False, "not_an_absolute_accuracy_certificate": True}
        return self.decision


def consume_until_stop(spec, frames: Iterable[ObservedFrame]):
    policy = OnlineStopRule(spec)
    for frame in frames:
        result = policy.observe(frame)
        if result is not None:
            return result
    return {"stop": False, "reason": "OBSERVED_STREAM_ENDED_NOT_RULE_CONVERGENCE", "observed_probes": len(policy.frames), "FCI_used": False}


def remaining_final_budget(total_raw_shots, used_pilot_raw_shots, minimum_final_shots=1):
    remaining = int(total_raw_shots)-int(used_pilot_raw_shots)
    if used_pilot_raw_shots < 0 or remaining < minimum_final_shots:
        raise ValueError("Pilot and independent final readout must share the registered total budget")
    return remaining
