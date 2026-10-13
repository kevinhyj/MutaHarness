"""Run a frozen predictor from supplied public support, then enforce the output contract."""
import argparse, importlib.util, json, math, sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--module', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    case = json.loads(a.case.read_text())
    ids = [r['id'] for r in case['query']]
    assignment = {'round': 3, 'arm': 'direct', 'batch_size': 0, 'eligible_ids': ids,
                  'audit_ids': ids, 'remaining_budget': 0}
    sys.path.insert(0, str(a.module.parent))
    spec = importlib.util.spec_from_file_location('frozen_predictor', a.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = module.propose(case, assignment)
    scores = output['scores']
    assert set(scores) == set(ids)
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and -1 <= v <= 1 for v in scores.values())
    assert output.get('selected_ids', []) == []
    a.out.write_text(json.dumps({'scores': scores, 'selected_ids': []}, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
