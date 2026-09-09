"""Offline Bash workflow regression tests; no dataset or models needed."""
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import unittest
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MOCK = r'''#!/usr/bin/env python3
import ast, csv, json, os, pathlib, sys
if sys.argv[1] == '-':
    sys.argv = sys.argv[1:]
    exec(compile(sys.stdin.read(), '<stdin>', 'exec'))
    raise SystemExit(0)
script, *args = sys.argv[1:]
tree = ast.parse(pathlib.Path(script).read_text())
allowed, required = set(), set()
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
        flags = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        allowed.update(flags)
        if any(k.arg == 'required' and isinstance(k.value, ast.Constant) and k.value.value for k in node.keywords):
            required.add(flags[0])
options = {}
for i, arg in enumerate(args):
    if arg.startswith('--'):
        assert arg in allowed, (script, 'unsupported option', arg)
        options[arg] = args[i+1] if i+1 < len(args) and not args[i+1].startswith('--') else True
assert required <= options.keys(), (script, 'missing', required - options.keys())
with open(os.environ['MOCK_LOG'], 'a') as log:
    log.write(json.dumps({'script': pathlib.Path(script).name, 'args': options})+'\n')
def write_csv(path, fields, rows):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
def cleanup(directory):
    p = pathlib.Path(directory)
    for row in csv.DictReader((p/'attack_batches.csv').open()):
        (p/row['file']).unlink(missing_ok=True)
name = pathlib.Path(script).stem
if name == 'visualize_frequency_spectrum':
    entries = [args[i+1] for i, arg in enumerate(args) if arg == '--attack-dir']
    assert len(entries) == 2 and '--attack-dirs' not in options, args
    for entry in entries:
        label, path = entry.split('=', 1)
        assert label and ',' in path and pathlib.Path(path).is_dir(), entry
if name == 'select_imagenet_subset':
    write_csv(options['--out-csv'], ['relpath','label'], [{'relpath':f'image_{i}.jpg','label':i} for i in range(int(options['--num-images']))])
elif name == 'run_attack':
    out = pathlib.Path(options['--out-dir'])
    batches = pathlib.Path(options['--adv-batch-dir'])
    out.mkdir(parents=True, exist_ok=True)
    batches.mkdir(parents=True, exist_ok=True)
    count = sum(1 for _ in csv.DictReader(open(options['--selected-csv'])))
    size = int(options['--batch-size'])
    limit = min((count+size-1)//size, int(options.get('--num-batches', '999999')))
    rows = []
    for index in range(limit):
        p = batches/f'batch_{index:05d}.pt'
        p.write_text('mock tensor')
        rows.append({'file': os.path.relpath(p,out), 'n':min(size,count-index*size)})
    write_csv(out/'attack_batches.csv', ['file','n'], rows)
elif name == 'evaluate':
    out = pathlib.Path(options['--attack-dir'])
    for row in csv.DictReader((out/'attack_batches.csv').open()):
        assert (out/row['file']).is_file(), row
    write_csv(out/'eval_results.csv', ['target','asr'], [{'target':target,'asr':1} for target in options['--targets'].split(',')])
    if options.get('--delete-batches-after-eval'):
        cleanup(out)
elif name == 'compute_perceptual_quality':
    manifest = list(csv.DictReader(open(options['--manifest'])))
    assert len(manifest) == 1 and None not in manifest[0], manifest
    write_csv(options['--out'], ['Family','Method','PSNR','SSIM','LPIPS','AttackDir'], [{'Family':r['family'],'Method':r['method'],'PSNR':30,'SSIM':1,'LPIPS':0,'AttackDir':r['attack_dir']} for r in manifest])
    if options.get('--delete-batches-after-use'):
        for row in manifest:
            cleanup(row['attack_dir'])
elif '--out' in options:
    p = pathlib.Path(options['--out'])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('offline mock output\n')
'''

def test_shell_workflows():
    if sys.platform == 'win32' or not shutil.which('bash'):
        raise unittest.SkipTest('Bash workflow smoke test requires Linux or macOS')
    with tempfile.TemporaryDirectory(prefix='shell-review-') as temp:
        tmp = Path(temp)
        mock = tmp/'python-mock'
        mock.write_text(MOCK)
        mock.chmod(0o755)
        log = tmp/'calls.jsonl'
        # Do not inherit real experiment/output settings from the invoking shell.
        env = {key: os.environ[key] for key in ('PATH', 'LANG', 'LC_ALL', 'TMPDIR') if key in os.environ}
        env.update(PY=str(mock), OUT_DIR=str(tmp/'output, with spaces'),
                   MOCK_LOG=str(log), NUM_IMAGES='2', BATCH_SIZE='2', NUM_BATCHES='1',
                   NUM_WORKERS='0', DEVICE='cpu', TABLE5P_NUM_IMAGES='2',
                   TABLE5P_VARIANTS='full_model', TABLE5P_SETTINGS='cnn_to_vit',
                   FIG_METHODS='mifgsm,ours', DEFENSE_METHODS='ours',
                   DEFENSE_TARGETS='resnet50', DELETE_ADV_AFTER_USE='1',
                   SKIP_TABLES1_4='1', SKIP_TABLE5='1', SKIP_TABLE6='1',
                   SKIP_TABLE7='1', SKIP_TABLE8='1')

        def run(name, overrides=None, expect=0):
            result = subprocess.run(['bash', str(ROOT/'sh'/name)], cwd=tmp,
                                    env=dict(env, **(overrides or {})), text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            assert result.returncode == expect, (name, result.returncode, result.stdout[-3000:], result.stderr[-3000:])
            return result

        def calls():
            return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

        scripts = sorted((ROOT/'sh').glob('*.sh'))
        for script in scripts:
            subprocess.run(['bash', '-n', str(script)], check=True)
            run(script.name)
        print(f'PASS bash syntax and external-cwd execution: {len(scripts)} scripts')

        before = len(calls())
        run('run_table8_defense.sh')
        assert any(r['script']=='run_attack.py' for r in calls()[before:]), 'deleted tensors reused'
        print('PASS deleted adversarial batches regenerate')

        before = len(calls())
        run('run_table5_prime.sh')
        assert not any(r['script']=='run_attack.py' for r in calls()[before:]), 'unchanged completed Table V did not reuse'
        before = len(calls())
        run('run_table5_prime.sh', {'SVFCA_BAND_TEMPERATURE':'0.5'})
        assert any(r['script']=='run_attack.py' for r in calls()[before:]), 'changed configuration reused stale results'
        print('PASS Table V reuses only matching configuration')

        before = len(calls())
        run('run_all_outputs.sh', {'RUN_FIGURES':'1','SKIP_TABLES1_4':'1','SKIP_TABLE5':'1',
                                 'SKIP_TABLE6':'1','SKIP_TABLE7':'1','SKIP_TABLE8':'1'})
        assert sum(r['script']=='visualize_attack_triplet.py' for r in calls()[before:]) == 1
        assert not list((Path(env['OUT_DIR'])/'adv_batches'/'figures').rglob('*.pt'))
        print('PASS figures run once and central tensors are deleted')

        run('common.sh', {'BATCH_SIZE':'0'}, expect=1)
        print('PASS invalid batch size fails early')

        # Test path containment and empty CSV opt-out without running any Python CLI.
        cmd = f'source "{ROOT}/sh/common.sh"; central_adv_batch_dir /tmp/external-experiment; printf "%s|%s\\n" "$LABELS_CSV" "$SVFCA_AMP"'
        result = subprocess.run(['bash','-c',cmd], env=dict(env,LABELS_CSV='',SVFTA_AMP='1'), text=True, capture_output=True, check=True)
        lines = result.stdout.splitlines()
        assert Path(lines[0]).is_relative_to(Path(env['OUT_DIR'])/'adv_batches')
        assert lines[1] == '|1', lines
        print('PASS external batch paths stay centralized; ImageFolder opt-out and legacy AMP alias')

        # The numeric filename wrapper must really select 5000 by default.
        selection_env = dict(env)
        selection_env.pop('NUM_IMAGES')
        selection_env.pop('NUM_BATCHES')
        before = len(calls())
        subprocess.run(['bash',str(ROOT/'sh'/'select_5000.sh')], cwd=tmp, env=selection_env,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        assert calls()[before]['args']['--num-images'] == '5000'
        print('PASS select_5000 selects 5000 images by default')


if __name__ == '__main__':
    test_shell_workflows()
