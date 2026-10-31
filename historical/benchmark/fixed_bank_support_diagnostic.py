"""Post-outcome, analytic-support diagnostic; never a practical CI-selection rule."""
import argparse
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from .common import ROOT,file_digest,save_json
from .local_sampler_primitives import LocalQE,unpack_counts64
from .resource_adaptive_sampler import integral_problem
from .sparse_qe_circuit import SparseQECircuit
from .fixed_bank_audit import SUCCESS


def analyze(root,probability_cutoff=1e-14):
    root=Path(root).resolve();cfg=json.loads((root/"manifest.json").read_text());validation=json.loads((root/"validation.json").read_text())
    assert validation["status"]==SUCCESS and file_digest(root/"events.sqlite")==validation["events_sha256"]
    out=root/"support-saturation-v1"
    if out.exists():raise RuntimeError("Additive diagnostic output required")
    db=sqlite3.connect(f"file:{root/'events.sqlite'}?mode=ro",uri=True);rows=[]
    for taskraw,resultraw in db.execute("SELECT task,result FROM bases WHERE status='COMPLETED' ORDER BY rowid"):
        task=json.loads(taskraw);result=json.loads(resultraw);case=result["case"]
        backend,info=integral_problem(Path(cfg["source"])/"references"/case)
        supports=[];omitted=[]
        for cid in result["shallow_members"]:
            actions=json.loads(db.execute("SELECT actions FROM circuits WHERE id=?",(cid,)).fetchone()[0])
            ops=[LocalQE(tuple(a["source"]),tuple(a["target"]),a["n_qubits"]) for a in actions]
            sim=SparseQECircuit(info["n_qubits"],info["hf_id"],ops,cfg["support_cap"])
            p=abs(sim.state([a["theta"] for a in actions]))**2
            supports.append(set(map(int,sim.ids[p>probability_cutoff]))|{info["hf_id"]});omitted.append(float(p[p<=probability_cutoff].sum()))
        union=set().union(*supports);last=supports[-1]
        # Ideal-distribution support is an outer diagnostic only. These IDs
        # are NEVER inserted into the frozen measured-only experiment.
        unionE=float(np.linalg.eigvalsh(backend.matrix(sorted(union),sorted(union)))[0])
        lastE=float(np.linalg.eigvalsh(backend.matrix(sorted(last),sorted(last)))[0])
        finalrecords={}
        for mode,rep,raw in db.execute("SELECT method,rep,record FROM finals WHERE base=?",(result["base_id"],)):
            f=json.loads(raw);observed=set()
            for did in f["component_draws"]:
                blob=db.execute("SELECT counts FROM draws WHERE id=?",(did,)).fetchone()[0];ids,_=unpack_counts64(blob);observed.update(map(int,ids))
            observed.add(info["hf_id"])
            finalrecords[(mode,rep)]={"energy":f["energy_Ha"],"observed":observed,"selected":set(f["selected_ids"])}
        uniform_delta=[1000*(finalrecords[("uniform",rep)]["energy"]-finalrecords[("single_last",rep)]["energy"]) for rep in range(cfg["repeats"])]
        full_support_fraction=np.mean([last<=finalrecords[("single_last",rep)]["observed"] for rep in range(cfg["repeats"])])
        rows.append({"case":case,"source_seed":result["source_seed"],"bank_members":len(supports),"union_support_above_cutoff":len(union),
                     "last_shallow_support_above_cutoff":len(last),"extra_union_configurations_vs_last":len(union-last),
                     "R_cap":cfg["R"],"R_covers_entire_diagnostic_union":len(union)<=cfg["R"],"maximum_omitted_probability":max(omitted),
                     "ideal_support_union_minus_last_Ritz_mHa":1000*(unionE-lastE),"single_last_fraction_of_final_draws_covering_last_support":float(full_support_fraction),
                     "uniform_minus_single_last_mean_mHa":float(np.mean(uniform_delta)),"uniform_minus_single_last_max_abs_mHa":float(max(abs(np.array(uniform_delta)))),
                     "analytic_probabilities_outer_diagnostic_not_used_by_policy":True})
    db.close();df=pd.DataFrame(rows);out.mkdir();df.to_csv(out/"per-source-support-diagnostic.csv",index=False)
    summary={"status":"POST_OUTCOME_ANALYTIC_SUPPORT_DIAGNOSTIC","bases":len(df),"probability_cutoff_assumption":probability_cutoff,
             "banks_with_any_extra_support_vs_last":int((df.extra_union_configurations_vs_last>0).sum()),
             "all_R_caps_cover_diagnostic_union":bool(df.R_covers_entire_diagnostic_union.all()),"largest_union_size":int(df.union_support_above_cutoff.max()),
             "maximum_omitted_probability":float(df.maximum_omitted_probability.max()),"largest_ideal_union_energy_gain_mHa":float(-df.ideal_support_union_minus_last_Ritz_mHa.min()),
             "maximum_uniform_last_replicate_difference_mHa":float(df.uniform_minus_single_last_max_abs_mHa.max()),
             "mean_last_support_observed_fraction":float(df.single_last_fraction_of_final_draws_covering_last_support.mean()),
             "source_events_sha256":validation["events_sha256"],"code_sha256":file_digest(Path(__file__)),"new_quantum_acquisitions":0,"new_FCI_solves":0,
             "practical_support_padding":False,"whole_study_complete":False}
    (out/"INTERPRETATION.md").write_text(f"""# Why this nested-prefix bank may not improve energy

This analysis was designed AFTER observing the allocation outcomes. It uses
ideal physical-circuit probabilities above a declared numerical cutoff
{probability_cutoff:g}, not only counts. It is an analytic diagnostic, NOT an
implementable configuration selector, ground-state oracle or new experiment.
No observed-only algorithm result or raw count was changed.

Across{len(df)}registered inherited banks, {summary['banks_with_any_extra_support_vs_last']}
have additional retained-probability configurations beyond the final shallow
member. Largest union size{summary['largest_union_size']}; R64 covers all these
diagnostic unions:{summary['all_R_caps_cover_diagnostic_union']}.
Maximum probability mass omitted by the cutoff:{summary['maximum_omitted_probability']:.3g}.
Mean fraction of last-only finaldraws covering its full diagnostic support:
{summary['mean_last_support_observed_fraction']:.6f}.
Maximum uniform-vs-last energy difference across paired replicate indices:
{summary['maximum_uniform_last_replicate_difference_mHa']:.6g}mHa.

If a prefix bank's union support equals its last member's support and finite
sampling already discovers that support withinR, reallocating shots has no
additional determinants available to improve the selected-space energy. It
can still reduce gate-shot execution proxies by using shallower members.
This is not a universal theorem about multishallow banks: genuinely different
operator topologies can have complementary support. Current evidence concerns
THIS registered nested-prefix construction, its M/R regime and numerical cutoff.
Do not call it failure of all multishallow sampling or claim new operator
discovery/diversity from duplicates of the same support.

The two small projected Ritz calculations per bank are outer diagnostics on
named circuit support, not FCI labels, and are not padded into measured QSCI.
No fullFock vector/Hamiltonian or newFCI routine is required. Whole study remains
incomplete; independent-topology banks would require a new explicit experiment.
""")
    save_json(out/"summary.json",summary);return summary


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root");a=p.parse_args();print(json.dumps(analyze(a.root)))
