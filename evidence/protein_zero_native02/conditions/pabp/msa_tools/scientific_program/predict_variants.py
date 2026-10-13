"""Label-free PAB1 variant ranking. No assay outcomes or external data used."""
import json, math, re, hashlib
from pathlib import Path
import numpy as np

AA='ARNDCQEGHILKMFPSTWYV'
# Standard BLOSUM62, a general biochemical substitution prior (not assay data).
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
[0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4]])
IX={a:i for i,a in enumerate(AA)}

def read_and_validate():
 c=json.load(open('case.json')); manifest=json.load(open('evidence_manifest.json'))
 e=json.load(open('msa_evidence.json')); prof=json.load(open('raw_evidence/msa_profile.json'))
 assert manifest['support_count']==0 and not manifest['measurements_present']
 for f,v in manifest['files'].items():
  if isinstance(v,dict) and 'sha256' in v:
   assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==v['sha256'],f
 wt=c['reference_sequence']
 assert e['reference_sequences_sha256']=={'__single__':hashlib.sha256(wt.encode()).hexdigest()}
 assert c['context']['higher_is_better'] is True
 assert not c.get('reference_chains') and not c.get('partners'), 'Explicit handling needed for partners'
 ids=[q['id'] for q in c['query']]
 assert len(ids)==250 and set(ids)=={f'M{i:03d}' for i in range(1,251)}
 assert e['candidate_order']==ids
 ei={x['id']:x for x in e['candidates']}
 for q in c['query']:
  seq=q['sequence']; assert len(seq)==len(wt)
  actual={(i,a,b) for i,(a,b) in enumerate(zip(wt,seq),1) if a!=b}
  declared=set()
  for token in q['mutation'].split(':'):
   mm=re.fullmatch(r'([A-Z])(\d+)([A-Z])',token); assert mm
   a,p,b=mm.groups(); declared.add((int(p),a,b))
  assert actual==declared and len(actual)==q['mutation_order'],q['id']
  terms=ei[q['id']]['substitution_terms']
  assert {(t['position'],t['WT'],t['ALT']) for t in terms}==actual
  for t in terms:
   p=prof['positions'][str(t['position'])]
   assert p['wt']==wt[t['position']-1]==t['WT']
   assert p['non_gap_effective_depth']==t['non_gap_depth']
   if t['non_gap_depth']<=0: assert t['sum_log_frequency_ratio'] is None
   else:
    assert math.isclose(t['sum_log_frequency_ratio'],math.log(p['frequency'][t['ALT']]/p['frequency'][t['WT']]),abs_tol=1e-12)
 return c,e,prof

def checkpoint(c,e):
 out=[]
 for q,x in zip(c['query'],e['candidates']):
  total=0.
  for t in x['substitution_terms']:
   w,a=IX[t['WT']],IX[t['ALT']]
   biochemical=0.5*(B62[w,a]-B62[w,w])
   lr=t['sum_log_frequency_ratio']
   # Absent evolution is imputed from a separate weak biochemical prior.
   total += 0.8*max(-9.,lr)+0.2*biochemical if lr is not None else 0.5*biochemical
  out.append({'id':q['id'],'score':round(math.tanh((total+4.)/8.),8)})
 return {'predictions':out,'summary':'Unsupervised checkpoint: independent-site homolog log-frequency ratios plus a weak BLOSUM62 substitution prior; absent homolog evidence uses the biochemical prior alone. Source hashes, WT mutation identities and complete IDs verified. Single global monotonic scale; no measured outcomes, structure or validated epistasis; scores are relative rankings, not calibrated assay values.'}

def final_predict(c,e,prof):
 # A BLOSUM62-derived biochemical kernel is a separate model feature;
 # the supplied frequencies and their fixed LR/LOR scores remain unchanged.
 # BLOSUM62 scores are in half-bit units; diagonal-relative distance gives
 # unit self-compatibility and lower compatibility for dissimilar residues.
 assert np.array_equal(B62,B62.T)
 diagonal=np.diag(B62)
 distance=(diagonal[:,None]+diagonal[None,:])/2-B62
 assert np.all(distance>=0)
 kernel=np.exp(-math.log(2)/2*distance)
 assert np.allclose(np.diag(kernel),1)
 output=[]; diagnostics=[]
 rnp=set(range(128,134)) | set(range(166,174))
 for q,x in zip(c['query'],e['candidates']):
  total=0.; terms=[]
  for t in x['substitution_terms']:
   pos,wt,alt=t['position'],t['WT'],t['ALT']
   iw,ia=IX[wt],IX[alt]
   lr=t['sum_log_frequency_ratio']
   soft=None
   if t['non_gap_depth']>0:
    f=np.array([prof['positions'][str(pos)]['frequency'][a] for a in AA])
    assert np.isclose(f.sum(),1)
    compatibility=kernel@f
    soft=float(np.log(compatibility[ia]/compatibility[iw]))
    # Fixed mixture, no fitted coefficients or phenotype-based selection.
    base=0.45*lr+0.55*soft
    evidence='observed_homolog_profile'
   else:
    # K164 is an unaligned insertion: use only a deliberately weak prior.
    base=0.20*float(B62[iw,ia]-B62[iw,iw])
    evidence='missing_profile_biochemical_prior_only'
   function_penalty=0.
   # Universal RRM RNP2/RNP1 aromatic stacking residues, mapped by sequence.
   # These are motif priors, not structure-derived contacts or affinity values.
   if pos in {129,168,170} and alt not in 'FYW':
    function_penalty += 0.5 if alt=='H' else 1.5
   if pos in {129,131,132,166,168,170,172} and alt in 'DE':
    function_penalty += 1.0
   if pos in {167,169} and alt not in 'GA':
    function_penalty += 0.8
   if pos in rnp and alt=='P':
    function_penalty += 1.0
   term=base-function_penalty
   total+=term
   terms.append({'position':pos,'WT':wt,'ALT':alt,'evidence':evidence,
                 'fixed_msa_log_ratio':lr,'biochemical_compatibility_log_ratio':soft,
                 'motif_penalty':function_penalty,'model_term':term})
  # One monotonic global mapping, with no position/block normalization.
  score=round(math.tanh((total+3.5)/6.),8)
  output.append({'id':q['id'],'score':score})
  diagnostics.append({'id':q['id'],'mutation':q['mutation'],
                      'msa_coverage':x['scores']['sum_log_frequency_ratio']['coverage'],
                      'raw_model_score':total,'terms':terms})
 result={'predictions':output,'summary':
  'Label-free additive ranking combines homolog log-frequency ratios, BLOSUM62 biochemical compatibility and modest RRM RNA-binding motif penalties. Source hashes, residue mapping, mutation identities and all 250 IDs verified. MSA coverage: 240 complete, 9 partial, 1 absent; K164 uses a weak biochemical prior, not a complete evolutionary score. One global scale; no structures or assay outcomes, no validated epistasis, and substantial uncertainty from growth versus natural conservation. Scores are relative, not calibrated assay values.'}
 return result,diagnostics

if __name__=='__main__':
 c,e,p=read_and_validate()
 if not Path('checkpoint_predictions.json').exists():
  Path('checkpoint_predictions.json').write_text(json.dumps(checkpoint(c,e),indent=2)+'\n')
 result,diagnostics=final_predict(c,e,p)
 ids=[x['id'] for x in result['predictions']]
 assert len(ids)==250 and len(set(ids))==250 and set(ids)=={f'M{i:03d}' for i in range(1,251)}
 assert all(math.isfinite(x['score']) and -1<=x['score']<=1 for x in result['predictions'])
 assert [x['id'] for x in result['predictions']]==[q['id'] for q in c['query']]
 Path('final_predictions.json').write_text(json.dumps(result,indent=2)+'\n')
 Path('prediction_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
 print('Saved and validated final_predictions.json: 250 unique IDs; range',min(x['score'] for x in result['predictions']),max(x['score'] for x in result['predictions']))
