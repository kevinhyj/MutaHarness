#!/usr/bin/env python3
"""Verify frozen artifact hashes and stored-result arithmetic; no predictor/LLM execution."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_viewer(manifest, pooled, zero):
    """Check the delivered UI data against frozen metrics and all raw predictions."""
    data = (ROOT/'demo/data.js').read_text()
    assert data.startswith('window.MUTA_DATA = ') and data.endswith(';\n')
    viewer = json.loads(data[len('window.MUTA_DATA = '):-2])
    for mode, feedback, queries in [('zero', 0, 250), ('ensemble', 96, 186)]:
        group = viewer[mode]
        assert group['feedback_count'] == feedback and group['query_count'] == queries
        assert len(group['cases']) == len({c['id'] for c in group['cases']}) == 10
        for method, value in group['means'].items():
            assert abs(value-statistics.mean(c['scores'][method] for c in group['cases'])) < 1e-12
    for case in viewer['zero']['cases']:
        for arm, value in case['scores'].items():
            row = next(r for r in zero if r['case_id']==case['id'] and r['arm']==arm)
            assert value == row['spearman']
        assert (ROOT/case['source']).is_file()
    for arm, value in viewer['zero']['top25_means'].items():
        assert abs(value-statistics.mean(r['top25_overlap'] for r in zero if r['arm']==arm)) < 1e-12
    n_candidates = 0
    for case in viewer['ensemble']['cases']:
        condition = case['id']
        input_case = read(ROOT/'inputs'/f'{condition}.json')
        row = next(r for r in pooled['rows'] if r['case_id']==manifest['conditions'][condition]['case_id'])
        assert case['scores'] == {'direct':row['candidate_rho'][0],
                                  'cv':row['cv_selected_single_rho'], 'ensemble':row['ensemble_rho']}
        assert case['cv_selected'] == row['cv_selected_candidate']
        raw = [read(ROOT/'reference_predictions'/condition/f'candidate_{k:02}.json')['scores'] for k in range(3)]
        candidates = {c['id']:c for c in case['candidates']}
        assert len(candidates) == len(case['candidates']) == 186
        assert set(candidates) == {q['id'] for q in input_case['query']}
        for query in input_case['query']:
            c = candidates[query['id']]
            assert c['mutation'] == query['mutation']
            scores = [p[query['id']] for p in raw]
            assert scores == c['program_scores']
            # Independent definition of the midrank percentile, including ties.
            votes = [(sum(v<s for v in p.values()) + .5*sum(v==s for v in p.values()))/len(p)
                     for p, s in zip(raw, scores)]
            assert abs(c['score']-(2*statistics.mean(votes)-1)) < 1e-12
            assert abs(c['disagreement']-statistics.pstdev(votes)) < 1e-12
            n_candidates += 1
    return n_candidates


def audit():
    manifest = read(ROOT/'artifacts_manifest.json')
    n_programs = 0
    for condition, details in manifest['conditions'].items():
        path = ROOT/'inputs'/f'{condition}.json'
        assert sha(path) == details['input_sha256']
        case = read(path)
        ids = {v['id'] for v in case['query']}
        assert len(case['support']) == 96 and len(ids) == 186
        assert all('fitness' not in v for v in case['query'])
        for i, candidate in enumerate(details['candidates']):
            for filename, digest in candidate['code_sha256'].items():
                assert sha(ROOT/'programs'/condition/f'candidate_{i:02}'/filename) == digest
            prediction = read(ROOT/'reference_predictions'/condition/f'candidate_{i:02}.json')
            assert set(prediction['scores']) == ids
            assert all(math.isfinite(v) and -1 <= v <= 1 for v in prediction['scores'].values())
            n_programs += 1
    pooled = read(ROOT/'evidence/scientific_program_ensemble/pooled_protein_metrics.json')
    for key in ['delta_vs_direct00', 'delta_vs_cv_single']:
        mean = statistics.mean(r[key] for r in pooled['rows'])
        assert abs(mean-pooled['comparisons'][key]['mean_delta']) < 1e-12
    zero = []
    for campaign in ['protein_zero_native01', 'protein_zero_native02']:
        summary = read(ROOT/'evidence'/campaign/'SUMMARY.json')
        assert summary['complete_conditions'] == summary['planned_conditions']
        assert summary['independent_arithmetic_verified']
        for row in summary['rows']:
            assert abs(row['spearman']-row['independent_spearman']) < 1e-12
        zero.extend(summary['rows'])
    means = {arm: statistics.mean(r['spearman'] for r in zero if r['arm']==arm)
             for arm in ['basic','msa_tools','structure_tools','joint_tools']}
    assert abs(means['basic']-0.5050867801038631) < 1e-12
    assert abs(means['joint_tools']-0.5735689527270961) < 1e-12
    n_viewer_candidates = audit_viewer(manifest, pooled, zero)
    return {'status':'passed', 'frozen_protein_conditions':10, 'programs_hash_verified':n_programs,
            'existing_zero_shot_conditions':len(zero), 'stored_metric_arithmetic_verified':True,
            'viewer_metric_values_verified':70, 'viewer_candidates_verified':n_viewer_candidates,
            'predictor_executions':0, 'new_model_calls':0, 'test_labels_opened':False,
            'scope':'hash/contract checks and stored-metric arithmetic; not fresh inference or full experiment reproduction'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out');args=p.parse_args()
    value=audit()
    if args.out: Path(args.out).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,indent=2))
