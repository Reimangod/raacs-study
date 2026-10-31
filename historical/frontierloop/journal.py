"""Compact raw-count journal. No per-candidate QPY/directory proliferation."""
from io import BytesIO
import json
from pathlib import Path
import sqlite3
from time import process_time,perf_counter
import numpy as np
from paperstudy.budget import BudgetLedger
from raacsnext.sampler import circuit_from_actions,provider
from .paths import ROOT
from .io import safe,space,arrays,digest,sha256

def normalized(actions):
    return [{'source':list(a['source']),'target':list(a['target']),'theta':float(a['theta'])}for a in actions]

def circuit_id(meta,actions):return digest(['physical-QE-actions',meta['hf_id'],meta['n_qubits'],normalized(actions)])

class Journal:
    def __init__(self,meta,directory,identity,budget,ledger_path=None):
        self.meta=meta;self.run_id=identity;self.directory=safe(Path(directory)/'draws.sqlite').parent
        self.directory.mkdir(parents=True,exist_ok=True)
        self.provider=provider(meta)
        self.ledger=BudgetLedger(safe(ledger_path or ROOT/'production-shots.sqlite'))
        self.ledger.register(identity,budget)
        self.db=sqlite3.connect(self.directory/'draws.sqlite',timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS draws(id TEXT PRIMARY KEY,request TEXT,counts BLOB,metadata TEXT,CPU_seconds REAL,wall_seconds REAL)')
        self.db.commit();self.new_draws=self.replayed_draws=0

    def measure(self,meta,actions,shots,role):
        if not isinstance(shots,int)or isinstance(shots,bool)or shots<=0:raise ValueError('Positive integer shots')
        if meta['problem_id']!=self.meta['problem_id']:raise ValueError('Cross-problem journal reuse')
        actions=normalized(actions);cid=circuit_id(meta,actions)
        request={'problem_id':meta['problem_id'],'circuit_id':cid,'actions':actions,'shots':shots,'role':role,
            'seed':str(int(digest([self.run_id,role])[:16],16))}
        request=json.loads(json.dumps(request));eid=digest(request);space()
        status=self.ledger.reserve(self.run_id,eid,request,shots)
        saved=self.db.execute('SELECT request,counts,metadata FROM draws WHERE id=?',(eid,)).fetchone()
        q=circuit_from_actions(meta,actions)
        if saved:
            if json.loads(saved[0])!=request:raise RuntimeError('Request drift')
            payload=saved[1];metadata=json.loads(saved[2])
            with np.load(BytesIO(payload),allow_pickle=False)as a:ids,counts=a['ids'],a['counts']
            self.ledger.complete(self.run_id,eid,int(counts.sum()),sha256(payload));self.replayed_draws+=1
        else:
            if status!='NEW_RESERVATION':raise RuntimeError('Unknown prior outcome; no blind remeasurement')
            tick,cpu=perf_counter(),process_time()
            ids,counts,metadata=self.provider.draw(q,shots,int(request['seed']),len(actions),len(actions))
            payload=arrays(ids=ids,counts=counts)
            self.db.execute('INSERT INTO draws VALUES(?,?,?,?,?,?)',(eid,json.dumps(request,sort_keys=True),payload,
                json.dumps(metadata,sort_keys=True),process_time()-cpu,perf_counter()-tick));self.db.commit()
            self.ledger.complete(self.run_id,eid,int(counts.sum()),sha256(payload));self.new_draws+=1
        return {'id':cid,'actions':actions,'observed':{int(x):int(n)for x,n in zip(ids,counts)},'metadata':metadata,'circuit':q}

    def cost(self):
        rows=self.db.execute('SELECT request,metadata,CPU_seconds,wall_seconds FROM draws').fetchall()
        return {'draws':len(rows),'sampling_CPU_seconds':sum(r[2]for r in rows),
            'sampling_wall_seconds':sum(r[3]for r in rows),
            'all_model_CNOT_applications':sum(json.loads(q)['shots']*json.loads(m)['sampling_resources']['cnot_count']for q,m,_,_ in rows)}

    def close(self):self.db.close()
