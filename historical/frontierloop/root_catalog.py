"""Context-bound CI observation reuse; old observations are never overwritten."""
import json
import sqlite3
from .io import digest
from .readout_seed_plan import evaluation_key


class Catalog:
    def __init__(self,path):
        self.db=sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE observations(seq INTEGER PRIMARY KEY,id TEXT UNIQUE,lookup TEXT,origin TEXT,record TEXT,cpu REAL,has_marginals INTEGER,is_new INTEGER)')
        self.db.execute('CREATE INDEX by_lookup ON observations(lookup,seq)')
        self.db.execute('CREATE TABLE attempts(seq INTEGER PRIMARY KEY,lookup TEXT,cell TEXT,status TEXT,cpu REAL,wall REAL,error TEXT)')
        self.db.commit()

    def put(self,context,row,origin,cpu,is_new=False):
        import numpy as np
        strings=row['CI_strings'];key=evaluation_key(context,strings)
        if 'spin_square' not in row and 'S2' in row:row={**row,'spin_square':row['S2']}
        if abs(row['energy_Ha']+.1*row['spin_square']-row['penalized_energy_Ha'])>1e-8:
            raise ValueError('Incompatible physical/penalized CI observation')
        existing=self.db.execute('SELECT record FROM observations WHERE lookup=? ORDER BY seq',(key,)).fetchall()
        for (record,) in existing:
            old=json.loads(record)
            if any(abs(old[k]-row[k])>1e-7 for k in ['energy_Ha','penalized_energy_Ha']):
                raise RuntimeError('Conflicting same-context CI roots; preserve and investigate')
        has='electron_marginal_weights_alpha' in row and 'electron_marginal_weights_beta' in row
        if has:
            for name in ['electron_marginal_weights_alpha','electron_marginal_weights_beta']:
                w=np.asarray(row[name])
                if w.shape!=(len(strings),) or np.any(w< -1e-14) or abs(w.sum()-1)>1e-8:
                    raise ValueError('Invalid endogenous CI marginal weights')
        oid=digest([key,origin,row])
        self.db.execute('INSERT OR IGNORE INTO observations(id,lookup,origin,record,cpu,has_marginals,is_new) VALUES(?,?,?,?,?,?,?)',
            (oid,key,origin,json.dumps(row,sort_keys=True,allow_nan=False),cpu,int(has),int(is_new)))
        self.db.commit();return key

    def find(self,key,need_marginals=False):
        sql='SELECT id,origin,record,cpu,is_new FROM observations WHERE lookup=?'
        if need_marginals:sql+=' AND has_marginals=1'
        r=self.db.execute(sql+' ORDER BY seq LIMIT 1',(key,)).fetchone()
        return None if r is None else dict(id=r[0],origin=r[1],record=json.loads(r[2]),cpu=r[3],is_new=bool(r[4]),key=key)

    def work(self,keys):
        """Uniform canonical CI proxy for a unique required-basis set, not cold time.

        A CI solve produces its eigenvector regardless of whether an old artifact
        retained it. Re-solving for missing historical marginals is charged to
        the NEW campaign, not pretended to be a second required standalone root.
        """
        records=[self.find(k) for k in sorted(set(keys))]
        if any(r is None for r in records):raise ValueError('Unknown required CI work')
        cpu=None if any(r['cpu'] is None for r in records) else sum(r['cpu'] for r in records)
        mv=[r['record'].get('matvec_calls') for r in records]
        return dict(unique_CI_evaluations=len(records),canonical_CI_CPU_proxy=cpu,
            canonical_matvec_proxy=None if any(x is None for x in mv) else sum(mv),
            canonical_observation_ids=[r['id'] for r in records],not_cold_end_to_end_time=True)

    def close(self):
        self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)');self.db.close()
