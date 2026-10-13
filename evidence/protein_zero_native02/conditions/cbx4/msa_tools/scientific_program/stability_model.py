"""Transparent unsupervised stability prior. All constants specified a priori.
Natural conservation can encode function; predictions are not measured ddG.
"""
import json,math
from pathlib import Path
import numpy as np
from predict import load_and_check,validate,generic_prior
ROOT=Path(__file__).resolve().parent
AA='ACDEFGHIKLMNPQRSTVWY'
# Standard BLOSUM62 half-bit substitution scores, in the order below.
ORDER='ARNDCQEGHILKMFPSTWYV'
BLOSUM62=np.array([
[4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0],
[-1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3],
[-2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3],
[-2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3],
[0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
[-1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2],
[-1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2],
[0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3],
[-2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3],
[-1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3],
[-1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1],
[-1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2],
[-1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1],
[-2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1],
[-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2],
[1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2],
[0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0],
[-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3],
[-2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1],
[0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4]],dtype=float)
assert np.array_equal(BLOSUM62,BLOSUM62.T)
ix=[ORDER.index(a) for a in AA]
B=BLOSUM62[np.ix_(ix,ix)]
K=np.exp(0.5*B);K/=K.sum(axis=1,keepdims=True)
HYDRO=set('ACVILMFWY')
VOLUME=dict(zip(AA,[89,109,111,138,190,60,153,167,169,167,163,114,113,144,173,89,116,140,228,194]))

def chemistry_cost(w,a,f):
    """Profile-supported hydrophobic compatibility, no claimed SASA/contacts."""
    hydromass=sum(f[x] for x in HYDRO)
    # Hydrophobic conservation is a weak packing proxy, not verified burial.
    core=max(0.0,min(1.0,(hydromass-0.5)/0.5))
    cost=0.0
    if w in HYDRO:
        if a in 'DEKR':cost+=2.5*core
        elif a not in HYDRO:cost+=1.2*core
        # Coarse side-chain size prior is symmetric, with no relaxed coordinates.
        cost+=0.6*core*abs(VOLUME[a]-VOLUME[w])/100.0
    # Backbone disruption is attenuated where homologs already favor G/P.
    flexible=min(1.0,(f['G']+f['P'])/0.25)
    if a=='P' and w!='P':cost+=1.4*(1-0.7*flexible)
    if a=='G' and w not in 'GP':cost+=0.45*(1-flexible)
    return cost,core

def predict():
    c,e,manifest=load_and_check()
    p=json.loads((ROOT/'resampled_msa_profile.json').read_text())
    assert p['msa_sha256']==manifest['files']['raw_evidence/CBX4_HUMAN_2023-08-07_b03.a2m']['sha256']
    assert p['reference_sha256']==e['reference_sequences_sha256']['__single__']
    rows=[];details=[]
    for q,x in zip(c['query'],e['candidates']):
        terms=[]
        for t in x['substitution_terms']:
            pos=t['position'];w=t['WT'];a=t['ALT'];site=p['positions'].get(str(pos))
            if site is None or site['non_gap_effective_depth']<=0:
                # Unsupported terminal insertions retain an explicit uncertain prior.
                effect=0.4*generic_prior(w,a)
                term={'position':pos,'WT':w,'ALT':a,'effect':effect,'profile_observed':False,'prior_only':True}
            else:
                assert site['wt']==w==c['reference_sequence'][pos-1]
                f=site['frequency'];fv=np.array([f[v] for v in AA])
                # Preserve the supplied/recomputed raw frequency in diagnostics;
                # the model's smoothed preference is a distinct derived prior.
                pref=0.75*fv+0.25*(fv@K)
                lr=math.log(pref[AA.index(a)]/pref[AA.index(w)])
                cost,core=chemistry_cost(w,a,f)
                # Conservation is moderated because selection includes function,
                # not just proteolysis stability. No fitted free parameters.
                effect=0.6*lr-0.35*cost
                term={'position':pos,'WT':w,'ALT':a,'effect':effect,'profile_observed':True,
                      'raw_profile_lr':math.log(f[a]/f[w]),'smoothed_preference_lr':lr,
                      'chemistry_cost':cost,'hydrophobic_proxy':core,
                      'effective_depth':site['non_gap_effective_depth']}
            terms.append(term)
        raw=math.fsum(t['effect'] for t in terms)
        rows.append({'id':q['id'],'score':round(math.tanh(raw/4.0),7)})
        details.append({'id':q['id'],'mutation':q['mutation'],'raw_score':raw,'terms':terms,
                        'supplied_msa_coverage':x['scores']['sum_log_frequency_ratio']['coverage']})
    validate(rows,c)
    summary=('Label-free additive stability prior from a reproducible 4,096-homolog sample across the supplied alignment, '
      'reweighted at 80% identity; this corrects the supplied lexical-prefix sampling coverage bias. '
      'BLOSUM62-smoothed preferences plus modest hydrophobic/size and backbone penalties distinguish conservative changes from disruptive ones. '
      'Hashes, WT mapping and all 250 mutation identities verified; residues 1/50 lack homolog evidence and use uncertain priors. '
      'One global tanh scale, higher is better. No outcomes or coordinates; conservation can reflect function, and epistasis/proteolysis artifacts remain unmodeled.')
    result={'predictions':rows,'summary':summary}
    (ROOT/'final_predictions.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'prediction_diagnostics.json').write_text(json.dumps(details,indent=2)+'\n')
    rationale={'checkpoint':'Observed fixed MSA marginal terms plus explicit generic priors for missing sites.',
      'refinement_1':'Raw-alignment audit showed supplied lexically selected first 2048 unique rows have zero coverage at WT 2..10, despite 60%-94% coverage across all unique homologs. Replace prefix with one seeded, uniform 4096-row sample and inverse-neighbor weights.',
      'refinement_2':'Stability and evolutionary function differ: pseudocount-only frequencies can assign equal penalties to hydrophobic and charged substitutions. Use fixed BLOSUM62 smoothing and small profile-supported hydrophobic, size, and proline/glycine priors. No geometric or validated epistatic claims.',
      'checks':'Manifest/source/reference hashes match; A2M query and 2..49 mapping verified; original MSA tool exactly reproduces supplied evidence; complete unique IDs, direction, finite bounds and mutations checked. These checks do not establish predictive accuracy.'}
    (ROOT/'model_rationale.json').write_text(json.dumps(rationale,indent=2)+'\n')
    print('Validated final predictions:',len(rows),'range',min(r['score'] for r in rows),max(r['score'] for r in rows))
    lookup={r['id']:r for r in rows}
    for d in sorted(details,key=lambda d:-d['raw_score'])[:15]+sorted(details,key=lambda d:d['raw_score'])[:10]:
        print(d['id'],d['mutation'],lookup[d['id']]['score'],[(t['position'],round(t['effect'],3)) for t in d['terms']])
if __name__=='__main__':predict()
