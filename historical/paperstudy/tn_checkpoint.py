"""Durable TN trial/best state, tied to an actual completed objective.

Call publish under the optimizer's lock. An initial operator alone cannot
become a valid best-state checkpoint. No surrogate is a final readout.
"""
from io import BytesIO
import hashlib,json,os
from pathlib import Path
import numpy as np


def sha(blob):return hashlib.sha256(blob).hexdigest()


def atomic(path,blob):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.pending')
    with tmp.open('wb')as f:f.write(blob);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def save(path,value):atomic(path,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode())


def arrays(path,values):
    stream=BytesIO();np.savez_compressed(stream,**values);blob=stream.getvalue();atomic(path,blob);return sha(blob)


def publish_best(root,config,record,theta,operator_arrays):
    root=Path(root);i=record['index']
    if record['status']!='COMPLETED' or not np.isfinite(record['CI']['energy_Ha']):raise ValueError('Actual completed finite objective required')
    if sha(np.asarray(theta,dtype='<f8').tobytes())!=record['theta_sha256']:raise ValueError('Trial coefficients do not match objective')
    if any(not np.isfinite(v).all()for v in operator_arrays.values()):raise ValueError('Nonfinite selected operator')
    trial=root/'surrogate-evaluations'/(str(i)+'.json')
    if not trial.exists()or json.loads(trial.read_text())!=record:raise ValueError('Completed trial must be durable first')
    destination=root/'checkpoints'/('best-'+str(i)+'.npz')
    op_hash=arrays(destination,operator_arrays)
    pointer={'status':'BOUND_BEST_FROM_COMPLETED_SURROGATE_NOT_FINAL_ENERGY','config_id':config['config_id'],
        'problem_id':config['problem_id'],'index':i,'surrogate_energy_Ha':record['CI']['energy_Ha'],
        'operator':str(destination.relative_to(root)),'operator_sha256':op_hash,
        'theta':np.asarray(theta).tolist(),'theta_sha256':record['theta_sha256'],
        'completed_trial':str(trial.relative_to(root)),'completed_trial_sha256':sha(trial.read_bytes()),
        'actual_QPU_shots':0,'independent_final_readout_still_required':True}
    save(root/'best-state.json',pointer);return pointer


def load_best(root,config):
    root=Path(root);pointer=json.loads((root/'best-state.json').read_text())
    if(pointer['config_id'],pointer['problem_id'])!=(config['config_id'],config['problem_id']):raise ValueError('Checkpoint belongs to another request')
    for name,key in(('operator','operator_sha256'),('completed_trial','completed_trial_sha256')):
        path=root/pointer[name]
        if root.resolve()not in path.resolve().parents or sha(path.read_bytes())!=pointer[key]:raise ValueError('Checkpoint artifact binding failed')
    record=json.loads((root/pointer['completed_trial']).read_text())
    if record['status']!='COMPLETED' or record['CI']['energy_Ha']!=pointer['surrogate_energy_Ha']:raise ValueError('No completed objective backing this checkpoint')
    if sha(np.asarray(pointer['theta'],dtype='<f8').tobytes())!=pointer['theta_sha256']:raise ValueError('Checkpoint theta changed')
    return pointer
