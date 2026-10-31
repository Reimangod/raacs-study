"""Recover native-exporter CPU from lossless closed compiler bundles.

One sequential archive pass, cached by archive SHA, rather than reopening a
tar for every member. No re-synthesis, sampling or scientific-data changes.
"""
import gzip,hashlib,json,tarfile
from collections import defaultdict
from .io import ROOT,save_json,sha256
from .analysis_products import write_csv


def inspect_bundle(archive):
    CPU=0.;wall=0.;count=0;membership=hashlib.sha256()
    with tarfile.open(archive,'r:gz')as tar:
        for member in tar:
            if not member.name.endswith('/member.json.gz'):continue
            report=json.loads(gzip.decompress(tar.extractfile(member).read()))
            CPU+=report['export_CPU_seconds'];wall+=report['export_wall_seconds'];count+=1
            membership.update(member.name.encode()+b'\n')
    return {'native_export_CPU_seconds':CPU,'native_export_wall_seconds_sum':wall,
        'compiled_member_records':count,'member_paths_digest':membership.hexdigest(),
        'covers_only_worker_timed_export_body':True,'new_synthesis_or_model_shots':0}


def run():
    out=ROOT/'analysis-v1/Rustiq-exporter-accounting';out.mkdir(parents=True,exist_ok=True)
    records=[];joins=list((ROOT/'rustiq-control-v1/common-readout-resource-joins').glob('*/*.json'))
    for join in joins:
        # Same archive naming rule as lossless_archive.closed_compiler_sweep.
        from .io import digest
        name='Rustiq-closed-'+digest(str(join.relative_to(ROOT)))
        manifest=ROOT/'raw-archives-v1'/(name+'.json')
        if not manifest.exists():continue # Known unaccounted work, not CPU0.
        m=json.loads(manifest.read_text());cache=out/(name+'.json')
        if cache.exists():
            value=json.loads(cache.read_text())
            if value['archive_sha256']!=m['archive_sha256']:raise RuntimeError('Compiler accounting source changed')
        else:
            archive=ROOT/m['archive']
            if sha256(archive.read_bytes())!=m['archive_sha256']:raise RuntimeError('Compiler archive bytes changed')
            value={**inspect_bundle(archive),'archive_sha256':m['archive_sha256'],'source_archive':m['archive'],
                'case_id':join.parent.name,'resource_join':str(join.relative_to(ROOT))}
            save_json(cache,value)
        records.append(value)
    native=sum(v['native_export_CPU_seconds']for v in records)
    parent=sum(json.loads(p.read_text())['local_CPU_seconds']for p in joins)
    setups=[json.loads(p.read_text())for p in(ROOT/'rustiq-control-v1/members').glob('*/group-preparations/*.json')]
    setup=sum(v['CPU_seconds']for v in setups)
    bycase=defaultdict(lambda:{'closed_bundles':0,'native_export_CPU_seconds':0.,'compiled_member_records':0})
    for v in records:
        c=bycase[v['case_id']];c['closed_bundles']+=1;c['native_export_CPU_seconds']+=v['native_export_CPU_seconds'];c['compiled_member_records']+=v['compiled_member_records']
    write_csv(ROOT/'analysis-v1/Rustiq-native-exporter-cost-by-case.csv',[{'case_id':k,**v}for k,v in sorted(bycase.items())])
    summary={'completed_primary_resource_joins':len(joins),'closed_bundles_accounted':len(records),
        'completed_joins_awaiting_lossless_archive':len(joins)-len(records),
        'native_export_CPU_seconds_known_subtotal':native,'native_grouping_setup_CPU_seconds_known':setup,
        'parent_controller_CPU_seconds_known':parent,'combined_disjoint_recorded_CPU_seconds_lower_bound':native+setup+parent,
        'worker_import_and_startup_CPU_not_fully_measured':True,
        'parent_controller_CPU_already_in_canonical_cost_evidence_do_not_add_again':True,
        'sum_of_wall_components_is_not_elapsed_campaign_time':True,'unarchived_or_live_export_CPU_not_assumed_zero':True,
        'new_model_shots':0,'new_synthesis_calls':0,'whole_experiment_complete':False}
    save_json(ROOT/'analysis-v1/Rustiq-native-exporter-cost-summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':run()
