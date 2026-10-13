import ast
from pathlib import Path
old=Path('proposal.py').read_text();dev=Path('model_dev.py').read_text(); tree=ast.parse(dev)
header='''"""Support-only deterministic ranking of GFP fluorescence variants.

Scientific assumptions and settings were compared on public support holdouts.
Every propose invocation fits anew; no data, model, or prediction file is read.
Returned scores are ordinal, not probabilities or calibrated assay predictions.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.special import expit, logit
from scipy.stats import rankdata
'''
constant=old[old.index("AA="):old.index('def features(')]
nodes=[]
for node in tree.body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['STRANDS','PARITIES'] for t in node.targets):nodes.append(ast.get_source_segment(dev,node))
 if isinstance(node,ast.FunctionDef) and node.name in ['annotation','featurize','target','kernel_pred','ge_pred']:nodes.append(ast.get_source_segment(dev,node))
ending='''

def propose(case, assignment):
    """Fit only supplied finite support fitness; score every current query."""
    reference = case['reference_sequence']
    query = case.get('query', [])
    if not query:
        return {'scores': {}, 'selected_ids': []}
    support = sorted([r for r in case.get('support', [])
                      if r.get('fitness') is not None
                      and np.isfinite(float(r['fitness']))],
                     key=lambda r: str(r.get('id', r.get('mutation', ''))))
    if len(support) < 3:
        pred = -np.array([len(mutations(r, reference)) for r in query], float)
    else:
        y = np.array([float(r['fitness']) for r in support])
        x, z, _ = featurize(support, reference)
        q, zq, _ = featurize(query, reference)
        components = []
        # Most weight goes to a reduced structural-prior model. The full
        # annotations have limited weight because ablations were mixed.
        # The ablation configuration is fixed, not learned support outcomes.
        removed = list(range(15, 19)) + list(range(20, 32))
        reduced_columns = [3+i for i in removed] + [44+i for i in removed] + [87, 88, 89]
        for use_full, block_weight in [(False, .75), (True, .25)]:
            xx, qq = x.copy(), q.copy()
            if not use_full:
                xx[:, reduced_columns] = 0.
                qq[:, reduced_columns] = 0.
            # At small training sizes, a strong local nonlinear kernel was
            # less stable. Deterministic size adaptation uses no held-out label.
            nonlinear_weight = .20 if len(support) >= 65 else .05
            weights = [.30, .30 + (.20-nonlinear_weight), nonlinear_weight, .20]
            preds = [
                kernel_pred(xx, qq, z, zq, y, alpha=.3, mode='raw', zw=.3),
                kernel_pred(xx, qq, z, zq, y, alpha=.3, mode='raw', zw=.3, rbf=1),
                kernel_pred(xx, qq, z, zq, y, alpha=.3, mode='raw', zw=.3, rbf=10),
                ge_pred(xx, qq, z, zq, y, alpha=.3, zw=.3, ceiling=4., floor=1.30),
            ]
            for weight, p in zip(weights, preds):
                # Query-wise ranks permit combining raw-assay and latent
                # predictions without pretending that their scales agree.
                p = np.nan_to_num(p, nan=float(np.median(y)), posinf=1e6, neginf=-1e6)
                components.append(block_weight * weight * (rankdata(p, method='average')-.5)/len(p))
        pred = np.sum(components, axis=0)
    scores = 2*(rankdata(pred, method='average')-.5)/len(pred)-1
    return {'scores': {r['id']: float(v) for r, v in zip(query, scores)},
            'selected_ids': []}
'''
Path('proposal.py').write_text(header+constant+'\n# Approximate sequence annotations, not measured structure.\n'+'\n\n'.join(nodes)+ending)
