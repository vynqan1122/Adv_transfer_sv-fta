"""Offline Bash workflow regression tests; no dataset or models needed."""
import json
import math
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

        # Every computational entry point must receive the same adaptive policy;
        # attack/runtime budgets must not silently fall back to CLI defaults.
        batch_flags = {'run_attack.py': '--batch-size',
                       'select_imagenet_subset.py': '--batch-size',
                       'evaluate.py': '--eval-batch-size',
                       'compute_perceptual_quality.py': '--quality-batch-size',
                       'measure_runtime.py': '--batch-size'}
        for call in calls():
            args = call['args']
            if call['script'] in batch_flags:
                assert args[batch_flags[call['script']]] == '2', call
                assert args['--min-batch-size'] == '1', call
                assert '--no-auto-batch' not in args, call
            if call['script'] in ('run_attack.py', 'measure_runtime.py'):
                assert math.isclose(float(args['--eps']), 16/255), call
                assert math.isclose(float(args['--alpha']), 1.6/255), call
                assert args['--steps'] == '10', call
                assert args['--consensus-gain'] == '2.0', call
                assert args['--energy-strength'] == '0.75', call
                assert args['--band-weight-floor'] == '0.02', call
        print('PASS common budgets and adaptive batch policy reach every CLI')

        before = len(calls())
        run('run_table8_defense.sh')
        assert any(r['script']=='run_attack.py' for r in calls()[before:]), 'deleted tensors reused'
        print('PASS deleted adversarial batches regenerate')

        before = len(calls())
        run('run_table5_ablation.sh')
        assert not any(r['script']=='run_attack.py' for r in calls()[before:]), 'unchanged completed Table V did not reuse'
        before = len(calls())
        run('run_table5_ablation.sh', {'SVFCA_BAND_TEMPERATURE':'0.5'})
        assert any(r['script']=='run_attack.py' for r in calls()[before:]), 'changed configuration reused stale results'
        print('PASS Table V reuses only matching configuration')

        for changed in ({'EPS': '4/255'}, {'ALPHA': '0.4/255'}, {'STEPS': '20'},
                        {'AUTO_BATCH': '0'}, {'MIN_BATCH_SIZE': '2'}):
            # Reset to the same base before testing each independent key.
            run('run_table5_ablation.sh')
            before = len(calls())
            run('run_table5_ablation.sh', changed)
            new_calls = calls()[before:]
            assert any(r['script'] == 'run_attack.py' for r in new_calls), changed
            if changed.get('AUTO_BATCH') == '0':
                for call in new_calls:
                    if call['script'] in ('run_attack.py', 'evaluate.py'):
                        assert call['args']['--no-auto-batch'] is True, call
        print('PASS budget and OOM policy changes invalidate cached results')

        before = len(calls())
        run('run_table8_defense.sh', {'EPS': '8/255', 'ALPHA': '0.8/255', 'STEPS': '11',
                                    'DEFENSE_EPS': '4/255', 'DEFENSE_ALPHA': '0.4/255',
                                    'DEFENSE_STEPS': '20'})
        defense_call = next(r for r in calls()[before:] if r['script'] == 'run_attack.py')
        assert math.isclose(float(defense_call['args']['--eps']), 4/255)
        assert math.isclose(float(defense_call['args']['--alpha']), 0.4/255)
        assert defense_call['args']['--steps'] == '20'
        print('PASS independent defense budget reaches Table VIII')

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


def test_config_preview():
    """Validate editable config/precedence without importing ML dependencies."""
    if sys.platform == 'win32' or not shutil.which('bash'):
        raise unittest.SkipTest('Bash configuration test requires Linux or macOS')
    with tempfile.TemporaryDirectory(prefix='config-review-') as temp:
        tmp = Path(temp)
        env = {key: os.environ[key] for key in ('PATH', 'LANG', 'LC_ALL', 'TMPDIR') if key in os.environ}
        env.update(PY=sys.executable, OUT_DIR=str(tmp/'not-created'))

        def preview(overrides=None):
            return subprocess.run(['bash', str(ROOT/'sh/common.sh')], cwd=tmp,
                                  env=dict(env, **(overrides or {})), text=True,
                                  capture_output=True)

        result = preview()
        assert result.returncode == 0, result.stderr
        assert '16 / 16 / 16' in result.stdout, result.stdout
        assert '1 / 1' in result.stdout, result.stdout
        assert not Path(env['OUT_DIR']).exists(), 'preview created output directories'

        overlay = tmp/'experiment.local.venv'
        overlay.write_text(': "${EPS:=4/255}"\n: "${ALPHA:=0.4/255}"\n'
                           ': "${STEPS:=20}"\n: "${BATCH_SIZE:=8}"\n'
                           ': "${NUM_BATCHES:=3}"\n: "${SKIP_TABLES1_4:=1}"\n'
                           ': "${SKIP_TABLE5:=1}"\n: "${SKIP_TABLE6:=1}"\n'
                           ': "${SKIP_TABLE7:=1}"\n: "${SKIP_TABLE8:=1}"\n')
        result = preview({'CONFIG_FILE': str(overlay)})
        assert result.returncode == 0, result.stderr
        assert '8 / 8 / 8' in result.stdout, result.stdout
        assert 'Selected images:        24' in result.stdout, result.stdout
        assert f'{4/255:.17g} / {float(2/1275):.17g} / 20' in result.stdout, result.stdout

        result = preview({'CONFIG_FILE': str(overlay), 'BATCH_SIZE': '16', 'STEPS': '7'})
        assert result.returncode == 0, result.stderr
        assert '16 / 16 / 16' in result.stdout, result.stdout
        assert 'Selected images:        48' in result.stdout, result.stdout
        assert f'{4/255:.17g} / {float(2/1275):.17g} / 7' in result.stdout, result.stdout

        # Parent launchers must load skip flags from the config file themselves.
        result = subprocess.run(['bash', str(ROOT/'sh/run_tables_1_8.sh')], cwd=tmp,
                                env=dict(env, CONFIG_FILE=str(overlay)), text=True,
                                capture_output=True)
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert result.stdout.count('[SKIP]') == 5, result.stdout
        assert not Path(env['OUT_DIR']).exists()

        result = preview({'EPS': '0', 'ALPHA': '0'})
        assert result.returncode == 0, result.stderr
        assert '0 / 0 / 10' in result.stdout, result.stdout

        for invalid in ({'EPS': '1/0'}, {'EPS': 'nan'}, {'EPS': '2'},
                        {'ALPHA': '-1/255'}, {'ALPHA': '1+2'}, {'STEPS': '0'},
                        {'BATCH_SIZE': '0'}, {'AUTO_BATCH': 'yes'},
                        {'MIN_BATCH_SIZE': '17'}, {'NUM_WORKERS': '-1'},
                        {'SVFCA_BAND_TEMPERATURE': '0'}, {'SVFCA_DECAY': 'inf'},
                        {'SVFCA_DIVERSITY_PROB': '2'}, {'SVFCA_SPECTRAL_DECAY': '-0.1'},
                        {'SVFCA_SPECTRAL_DECAY': '1'}, {'SVFCA_SPECTRAL_BANDS': '1'},
                        {'SVFCA_BAND_WEIGHT_FLOOR': '0.2'},
                        {'CONFIG_FILE': str(tmp/'missing.venv')}):
            result = preview(invalid)
            assert result.returncode != 0, invalid
            assert '[config]' in result.stderr, (invalid, result.stderr)
        print('PASS standalone config preview, defaults, overlays, precedence and early validation')


if __name__ == '__main__':
    test_config_preview()
    test_shell_workflows()
