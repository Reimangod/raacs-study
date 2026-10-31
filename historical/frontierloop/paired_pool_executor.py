"""Auditable mechanical successor of the unchanged qualified N7 executor."""


def generate(base):
    changes=[
        ('def native(folder,fcidump,nelec,strings):','def native(folder,fcidump,nelec,strings,initial_pairs=None):'),
        ("        for spin in ['Alpha','Beta']:(folder/(spin+'Dets.bin')).write_bytes(b''.join(int(x).to_bytes(16,'big') for x in strings))",
         "        for spin,index in [('Alpha',0),('Beta',1)]:\n"
         "            values=strings if initial_pairs is None else [p[index] for p in initial_pairs]\n"
         "            (folder/(spin+'Dets.bin')).write_bytes(b''.join(int(x).to_bytes(16,'big') for x in values))\n"
         "        native_env=os.environ.copy();native_env.pop('CEO_DICE_PAIRED_INPUT_V1',None)\n"
         "        if initial_pairs is not None:native_env['CEO_DICE_PAIRED_INPUT_V1']='1'"),
        ("p=subprocess.Popen([spec['native_binary']],cwd=folder,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,",
         "p=subprocess.Popen([spec['native_binary']],cwd=folder,env=native_env,stdout=log,stderr=subprocess.STDOUT,"),
        ("n,pairs,native_record=native(folder,definition['fcidump_path'],tuple(definition['nelec']),plan['initial_strings'])",
         "n,pairs,native_record=native(folder,definition['fcidump_path'],tuple(definition['nelec']),plan['initial_strings'],plan.get('initial_pairs'))"),
        ("new_model_shots=0,new_QPU_shots=0)\n        if plan['source']['status']",
         "new_model_shots=0,new_QPU_shots=0,initialization_arm=plan['initialization_arm'])\n        if plan['source']['status']")]
    for before,after in changes:
        if base.count(before)!=1:raise ValueError('Qualified executor context drift; no fuzzy change')
        base=base.replace(before,after)
    return base
