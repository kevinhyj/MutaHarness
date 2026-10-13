"""Family-specific, label-free UBE4B stability prior.

Raw A2M alignment: 50% WT identity across 64 match columns and >=85%
canonical-AA coverage; exact deduplication; inverse 80%-identity-neighbor
weighting. A fixed 10% BLOSUM62 conditional substitution mixture regularizes
rare alternatives. This constructs a SEPARATE raw-MSA profile; supplied profile
frequencies and fixed-tool outputs are unchanged. No fitted parameters or labels.
"""
import json,math,hashlib
from pathlib import Path
import numpy as np
from predict import load_validate,ROOT

AA='ARNDCQEGHILKMFPSTWYV'
B62=np.array([
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
BACKGROUND=np.array([.074,.052,.045,.054,.025,.034,.054,.074,.026,.068,.099,.058,.025,.047,.039,.057,.051,.013,.032,.073])
BACKGROUND/=BACKGROUND.sum()
TRANSITION=BACKGROUND[None,:]*np.exp2(B62/2)
TRANSITION/=TRANSITION.sum(axis=1,keepdims=True)


def read_close_alignment(wt):
    path=ROOT/'raw_evidence/UBE4B_HUMAN_2023-08-07_b04.a2m'
    # First record is the supplied WT query (mouse matches the given domain).
    with open(path) as f:
        header=next(f).strip();query_raw=next(f).strip()
    ungapped=''.join(a.upper() for a in query_raw if a.isalpha())
    assert ungapped==wt,'Raw query does not map exactly to the case reference'
    positions=[p for p,a in enumerate(query_raw,1) if a.isupper()]
    assert positions==list(range(4,68))
    query=''.join(a for a in query_raw if a.isupper())
    assert query==wt[3:67]
    strings=set();nraw=0;nretained=0
    with open(path) as f:
        for line in f:
            if line.startswith('>'):continue
            s=''.join(a for a in line.strip() if a.isupper() or a=='-')
            assert len(s)==64
            nraw+=1
            coverage=sum(a in AA for a in s)/64
            identity=sum(a==b for a,b in zip(s,query))/64
            if coverage>=.85 and identity>=.5:
                strings.add(s);nretained+=1
    strings=sorted(strings)
    enc={a:i for i,a in enumerate(AA)}
    arr=np.array([[enc.get(a,20) for a in s] for s in strings],dtype=np.uint8)
    weights=np.empty(len(arr))
    for start in range(0,len(arr),64):
        block=arr[start:start+64]
        common=(block[:,None,:]<20)&(arr[None,:,:]<20)
        depth=common.sum(axis=2)
        matches=((block[:,None,:]==arr[None,:,:])&common).sum(axis=2)
        neighbors=((depth>=32)&(matches>=.8*depth)).sum(axis=1)
        assert np.all(neighbors>=1)
        weights[start:start+len(block)]=1/neighbors
    counts=np.array([[weights[arr[:,col]==a].sum() for a in range(20)] for col in range(64)])
    depths=counts.sum(axis=1)
    frequencies=counts/depths[:,None]
    regularized=.9*frequencies+.1*(frequencies@TRANSITION)
    assert np.allclose(regularized.sum(axis=1),1)
    meta={'raw_sequence_count':nraw,'retained_raw_count':nretained,'unique_count':len(arr),
          'neff':float(weights.sum()),'minimum_site_effective_depth':float(depths.min()),
          'maximum_site_effective_depth':float(depths.max()),'identity_threshold':.5,
          'canonical_coverage_threshold':.85,'redundancy_identity_threshold':.8,
          'substitution_prior_fraction':.1,'match_column_reference_positions':positions,
          'absent_reference_positions':[1,2,3,68,69],
          'mapping':'First raw query exactly matches full case WT; lowercase query insertions excluded.',
          'raw_msa_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    return positions,frequencies,regularized,depths,meta


def main():
    case,manifest,side=load_validate()
    positions,freq,smooth,depths,meta=read_close_alignment(case['reference_sequence'])
    index={p:i for i,p in enumerate(positions)}
    predictions=[];details=[]
    for q,e in zip(case['query'],side['candidates']):
        terms=[]
        for t in e['substitution_terms']:
            w,a=AA.index(t['WT']),AA.index(t['ALT']);p=t['position']
            if p in index:
                j=index[p]
                value=math.log(smooth[j,a]/smooth[j,w])
                term={'position':p,'WT':t['WT'],'ALT':t['ALT'],'term':value,'evolutionary_evidence':True,
                      'effective_depth':float(depths[j]),'observed_frequency_WT':float(freq[j,w]),
                      'observed_frequency_ALT':float(freq[j,a]),'regularized_frequency_WT':float(smooth[j,w]),
                      'regularized_frequency_ALT':float(smooth[j,a])}
            else:
                # Terminal insertions have no homolog mapping. Weak generic prior:
                # slightly adverse for a change, increasingly adverse for chemically
                # dissimilar substitutions. It is not an evolutionary observation.
                value=-.15-.10*max(0,-B62[w,a])
                term={'position':p,'WT':t['WT'],'ALT':t['ALT'],'term':value,
                      'evolutionary_evidence':False,'effective_depth':None}
            terms.append(term)
        raw=math.fsum(t['term'] for t in terms)
        score=math.tanh(raw/8.)
        predictions.append({'id':q['id'],'score':score})
        details.append({'id':q['id'],'mutation':q['mutation'],'raw':raw,
                        'coverage':sum(t['evolutionary_evidence'] for t in terms)/len(terms),'terms':terms})
    assert len(predictions)==250 and len({p['id'] for p in predictions})==250
    assert all(math.isfinite(p['score']) and -1<=p['score']<=1 for p in predictions)
    summary=(f"Unsupervised stability prior from the verified raw MSA: {meta['unique_count']} unique homologs with >=50% WT identity and >=85% coverage, inverse-neighbor weighting (Neff {meta['neff']:.1f}), and a fixed 10% BLOSUM62 substitution prior. Additive log-frequency ratios use one global tanh scale; higher is better. This avoids the broad supplied profile's cysteine-rich family mixture. Residues 1–3 and 68–69 lack aligned evidence and use a weak physicochemical prior. Hashes and all mutation identities verified. No assay labels, structure or validated epistasis; natural function and lineage can confound stability.")
    result={'predictions':predictions,'summary':summary}
    json.dump(result,open(ROOT/'final_predictions.json','w'),indent=2,allow_nan=False)
    json.dump({'metadata':meta,'candidates':details},open(ROOT/'final_diagnostics.json','w'),indent=2,allow_nan=False)
    json.dump({'format':'separate_raw_alignment_profile_v1','metadata':meta,'AA_order':AA,
               'positions':positions,'observed_frequencies':freq.tolist(),
               'regularized_frequencies':smooth.tolist(),'site_effective_depths':depths.tolist()},
              open(ROOT/'family_profile.json','w'),indent=2,allow_nan=False)
    print(json.dumps(meta,indent=2))
    print('Final score range',min(p['score'] for p in predictions),max(p['score'] for p in predictions))
    ranked=sorted(zip(case['query'],predictions),key=lambda x:x[1]['score'],reverse=True)
    print('Top 12',[(q['id'],q['mutation'],round(p['score'],3)) for q,p in ranked[:12]])
    print('Bottom 12',[(q['id'],q['mutation'],round(p['score'],3)) for q,p in ranked[-12:]])

if __name__=='__main__':main()
