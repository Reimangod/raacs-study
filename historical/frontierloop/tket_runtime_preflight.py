"""N36 small private-wheel feasibility; never changes a shared environment."""
import json
import urllib.request
import zipfile
import subprocess
import shutil
from io import BytesIO
from time import process_time, perf_counter
from .paths import ROOT
from .io import save, sha256, space, load, write

RUN = ROOT / 'runs/N36-TKET-QUALIFICATION-v1'
RUNTIME = ROOT / 'runtime/N36-pytket-v1'
PINS = {'pytket': '2.18.5', 'lark': '1.3.1', 'graphviz': '0.21',
        'jinja2': '3.1.6', 'qwasm': '1.0.1', 'markupsafe': '3.0.3'}


def main():
    if (RUN / 'wheel-preflight.json').exists():
        raise RuntimeError('Do not replace the recorded package preflight')
    cpu, wall = process_time(), perf_counter()
    space()
    records = []
    for name, ver in PINS.items():
        with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{ver}/json', timeout=30) as r:
            metadata = json.load(r)
        wheels = [x for x in metadata['urls'] if x['filename'].endswith('.whl') and
                  (x['filename'].endswith('none-any.whl') or
                   ('cp311-cp311-macosx' in x['filename'] and 'arm64' in x['filename']))]
        if len(wheels) != 1:
            raise ValueError((name, [x['filename'] for x in wheels]))
        wheel = wheels[0]
        with urllib.request.urlopen(wheel['url'], timeout=30) as r:
            blob = r.read()
        if sha256(blob) != wheel['digests']['sha256'] or len(blob) != wheel['size']:
            raise ValueError('Wheel digest mismatch')
        with zipfile.ZipFile(BytesIO(blob)) as z:
            expanded = sum(x.file_size for x in z.infolist())
            license_files = [x.filename for x in z.infolist()
                             if 'license' in x.filename.lower() or 'licence' in x.filename.lower()]
        records.append({'name': name, 'version': ver, 'url': wheel['url'],
                        'SHA256': sha256(blob), 'compressed_bytes': len(blob),
                        'expanded_bytes': expanded, 'license_files': license_files,
                        'metadata_license': metadata['info'].get('license'),
                        'metadata_license_expression': metadata['info'].get('license_expression'),
                        'requires_dist': metadata['info'].get('requires_dist')})
    total = sum(x['expanded_bytes'] for x in records)
    if total > 28 * 2**20:
        raise OSError('32MiB private runtime budget cannot accommodate wheels plus metadata')
    result = {'status': 'SMALL_PRIVATE_RUNTIME_FEASIBLE_NOT_INSTALLED',
              'packages': records, 'expanded_wheel_bytes': total,
              'runtime_max_bytes': 32 * 2**20, 'runtime': str(RUNTIME),
              'CPU_seconds': process_time()-cpu, 'wall_seconds': perf_counter()-wall,
              'shared_environment_changed': False, 'new_model_shots': 0}
    save(RUN / 'wheel-preflight.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['install']:
        spec = load(RUN / 'wheel-preflight.json')
        if RUNTIME.exists():
            raise RuntimeError('Private runtime already exists; do not reinstall')
        space()
        cpu, wall = process_time(), perf_counter()
        with urllib.request.urlopen('https://raw.githubusercontent.com/Quantinuum/tket/main/LICENSE', timeout=30) as r:
            license_text = r.read()
        if b'Apache License' not in license_text or b'Version 2.0' not in license_text:
            raise ValueError('Public license not verified')
        write(RUN / 'tket-Apache-2.0-LICENSE.txt', license_text)
        command = [shutil.which('uv'), 'pip', 'install', '--no-cache', '--no-deps',
                   '--no-compile-bytecode', '--target', str(RUNTIME)]
        command += [x['url']+'#sha256='+x['SHA256'] for x in spec['packages']]
        child = subprocess.run(command, capture_output=True, text=True, timeout=120)
        files = {str(p.relative_to(RUNTIME)): sha256(p.read_bytes())
                 for p in sorted(RUNTIME.rglob('*')) if p.is_file()}
        total = sum(p.stat().st_size for p in RUNTIME.rglob('*') if p.is_file())
        receipt = {'returncode': child.returncode, 'command': command,
                   'stdout': child.stdout, 'stderr': child.stderr,
                   'wheel_preflight_SHA256': sha256((RUN/'wheel-preflight.json').read_bytes()),
                   'license_SHA256': sha256(license_text), 'installed_bytes': total,
                   'installed_files_SHA256': files, 'CPU_seconds_parent': process_time()-cpu,
                   'wall_seconds': perf_counter()-wall, 'shared_environment_changed': False}
        save(RUN / 'installation-receipt.json', receipt)
        print(json.dumps({k:v for k,v in receipt.items() if k != 'installed_files_SHA256'}))
        if child.returncode or total > spec['runtime_max_bytes']:
            raise RuntimeError('Private package installation failed or exceeded cap; receipt preserved')
    else:
        main()
