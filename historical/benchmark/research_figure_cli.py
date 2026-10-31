"""Opt-in full-study figure driver; preserves all legacy plot commands.

Regenerates no completed experiment. Existing completed derivative builders
validate their hashes, missing derivative outputs are built from saved inputs.
Produces an honest18-form index, not a fake complete Fig18.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def build(output,mode='standard'):
    from .final_relationship_figures import build as relationships
    from .final_relationship_followup import run as validate_relationships
    from .final_cost_frontiers_v2 import build as cost
    from .final_cost_frontier_audit import audit as validate_cost
    from .research_figure_inventory import build as inventory
    rroot=ROOT/f'results/final-relationships-{mode}-v1'
    croot=ROOT/f'results/final-cost-frontiers-{mode}-v2'
    relationships(rroot,mode);validate_relationships(rroot)
    cost(croot,mode);validate_cost(croot)
    inv=inventory();root=Path(inv['snapshot']);data=json.loads((root/'inventory.json').read_text())
    # This index covers the actual STANDARD cohort; quick qualifies builders,
    # not the scientifically independent30-condition outcomes.
    record={'schema':'scalable-study-available-figure-driver-v1','mode':mode,
        'completed_derivative_roots_checked':[str(rroot),str(croot)],
        'standard_figure_inventory':inv,'new_draws':0,'new_solver_calls':0,'hardware_shots':0,
        'whole_goal_complete':False,'all18forms_complete':not inv['still_missing_required_forms'],
        'no_partial_cost_substituted_for_Fig18':True,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inventory_sha256':hashlib.sha256((root/'inventory.json').read_bytes()).hexdigest()}
    body=json.dumps(record,sort_keys=True,indent=2)+'\n';identity=hashlib.sha256(body.encode()).hexdigest()
    dest=Path(output)/'snapshots'/identity;dest.mkdir(parents=True,exist_ok=True)
    lines=['# Scalable QSCI/SQD study — actual available figures','',
        'An available file is not a final scientific completion certificate. Missing forms remain missing.',
        f'Builder qualification mode:{mode}. Figure inventory below refers to the standard cohort.','',
        '|Requested form|Available3-format sets|Scope|','|---|---:|---|']
    for r in data['rows']:
        lines.append(f'|{r["figure"]}: {r["requested_title"]}|{len(r["candidate_pngs"])}|{r["scope"]}|')
        if r['candidate_pngs']:
            png=ROOT/r['candidate_pngs'][0]
            lines.append('\nRepresentative: '+' / '.join(f'[{ext.upper()}]({os.path.relpath(png.with_suffix("."+ext),dest)})' for ext in ('png','svg','pdf'))+'\n')
    lines+=['','Missing required forms: '+str(inv['still_missing_required_forms']),
        'No new sampling/state preparation/optimization/CI solve was performed. Hardware shots0.']
    for name,text in [('driver.json',body),('INDEX.md','\n'.join(lines)+'\n')]:
        p=dest/name
        if p.exists():assert p.read_text()==text
        else:p.write_text(text)
    (Path(output)/'LATEST.md').write_text(f'# Latest actual-figure index\n\n[18-form index](snapshots/{identity}/INDEX.md)\n\nWhole study remains incomplete.\n')
    return {'snapshot':str(dest),'requested_forms':18,'candidate_forms':inv['candidate_types'],
        'missing_forms':inv['still_missing_required_forms'],'whole_goal_complete':False,'new_draws':0}


def main():
    p=argparse.ArgumentParser();p.add_argument('--all',action='store_true')
    p.add_argument('--mode',choices=['quick','standard'],default='standard')
    p.add_argument('--output',default=str(ROOT/'results/research-figure-driver-v1'));a=p.parse_args()
    print(json.dumps(build(a.output,a.mode),sort_keys=True))


if __name__=='__main__':main()
