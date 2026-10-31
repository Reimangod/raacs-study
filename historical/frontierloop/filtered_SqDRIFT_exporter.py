"""N57 isolated SDK 2.5 exporter. Changes only the registered source operator."""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
from time import process_time,perf_counter
import numpy as np
from paperstudy.sqdrift_native_worker import Exporter
from qiskit_fermions.operators.terms.filtering import filter_diagonal_terms


class FilteredExporter(Exporter):
    def __init__(self,request):
        super().__init__(request);self.full_lambda=float(sum(self.weights))
        filter_diagonal_terms(self.operator);self.weights=self.operator.group_weights()
        self.first_moment_error=max(abs(self.weights[i]*np.sign(c)-c)
            for i,g in enumerate(self.operator.split_out_groups()) for _,c in g.iter_terms())
        if self.first_moment_error>1e-10:raise ValueError('Filtered group normalization differs')

    def export(self,spec,target):
        result=super().export(spec,target)
        path=target/'member.json.gz';r=json.loads(gzip.decompress(path.read_bytes()))
        r.update(diagonal_terms_retained=False,
            adaptation='N57 native SDK offdiagonal source heuristic; NOT full-H evolution or distribution-preserving rewrite',
            filtered_group_weight_sum=float(sum(self.weights)),original_group_weight_sum=self.full_lambda)
        blob=(json.dumps(r,sort_keys=True)+'\n').encode();path.write_bytes(gzip.compress(blob,mtime=0))
        result['member_sha256']=hashlib.sha256(blob).hexdigest();return result


def serve(base,directory):
    directory=Path(directory).resolve();cpu,wall=process_time(),perf_counter()
    request=json.loads(Path(base).read_text());exporters={'FULL_N20':Exporter(request),'FILTERED_SHORT':FilteredExporter(request)}
    print(json.dumps(dict(ready=True,setup_CPU_seconds=process_time()-cpu,setup_wall_seconds=perf_counter()-wall,
        full_lambda=float(sum(exporters['FULL_N20'].weights)),filtered_lambda=float(sum(exporters['FILTERED_SHORT'].weights)))),flush=True)
    for line in sys.stdin:
        r=json.loads(line)
        if r.get('stop'):return
        target=(directory/r['relative_directory']).resolve()
        if directory not in target.parents:raise ValueError('New namespace only')
        target.mkdir(parents=True,exist_ok=True)
        if (target/'member.json.gz').exists():raise RuntimeError('Existing member; do not re-export')
        result=exporters[r['arm']].export(r['spec'],target)
        result.update(worker_CPU_seconds=process_time()-cpu,worker_wall_seconds=perf_counter()-wall)
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base');p.add_argument('--directory');a=p.parse_args();serve(a.base,a.directory)
