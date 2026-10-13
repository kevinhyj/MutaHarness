"""Cold replay of the three frozen scientific programs; no model calls or evaluator reads."""
import argparse, csv, hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', default='gfp_s2501')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    manifest = read(ROOT / 'artifacts_manifest.json')
    assert a.case in manifest['conditions'], 'Unknown condition'
    if a.out.exists():
        raise ValueError('Output exists; choose a new directory')
    a.out.mkdir(parents=True)
    expected = manifest['conditions'][a.case]
    input_path = ROOT / 'inputs' / f'{a.case}.json'
    assert sha(input_path) == expected['input_sha256']
    case = read(input_path)
    ids = [r['id'] for r in case['query']]
    assert len(case['support']) == 96 and len(ids) == 186
    assert all('fitness' not in r for r in case['query'])
    predictions = []
    with tempfile.TemporaryDirectory(prefix='programportfolio-') as temp:
        for k in range(3):
            source = ROOT / 'programs' / a.case / f'candidate_{k:02}'
            ws = Path(temp) / f'candidate_{k:02}'
            ws.mkdir()
            for rel, digest in expected['candidates'][k]['code_sha256'].items():
                assert sha(source / rel) == digest
                target = ws / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / rel, target)
            output = a.out.resolve() / f'candidate_{k:02}.json'
            env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                   'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1'}
            subprocess.run([sys.executable, str(ROOT / 'runtime/replay_one.py'),
                            '--case', str(input_path), '--module', str(ws / 'proposal.py'), '--out', str(output)],
                           cwd=ws, env=env, timeout=120, check=True)
            value = read(output)
            assert value == read(ROOT / 'reference_predictions' / a.case / f'candidate_{k:02}.json'), 'Stored predictions did not reproduce exactly'
            predictions.append(value['scores'])
            for rel, digest in expected['candidates'][k]['code_sha256'].items():
                assert sha(ws / rel) == digest, 'Scientific code changed during execution'
    votes = np.array([(rankdata([v[i] for i in ids]) - .5) / len(ids) for v in predictions])
    fused = 2 * votes.mean(0) - 1
    out = {'scores': {i: float(fused[j]) for j, i in enumerate(ids)}, 'selected_ids': []}
    (a.out / 'ensemble.json').write_text(json.dumps(out, indent=2) + '\n')
    with (a.out / 'ranking.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'ensemble_score', 'program00_score', 'program01_score', 'program02_score', 'rank_disagreement_sd'])
        for j in sorted(range(len(ids)), key=lambda j: (-fused[j], ids[j])):
            i = ids[j]
            w.writerow([i, fused[j], *(v[i] for v in predictions), float(votes[:, j].std())])
    (a.out / 'verification.json').write_text(json.dumps({'condition': a.case, 'programs': 3,
        'all_program_hashes_verified': True, 'predictions_match_stored_results': True,
        'support_count': 96, 'query_count': 186, 'new_model_calls': 0,
        'evaluator_labels_opened': False}, indent=2) + '\n')
    print('PASS: all3 frozen programs refitted; predictions exactly reproduced; no model calls or evaluator labels.')


if __name__ == '__main__':
    main()
