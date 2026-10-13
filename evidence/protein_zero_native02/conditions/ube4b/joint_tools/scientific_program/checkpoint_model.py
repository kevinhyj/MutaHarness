"""Label-free stability prior; no assay supervision. Reproducible saved program."""
import json, math, re, hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
load=lambda n: json.loads((ROOT/n).read_text())
case=load('case.json'); manifest=load('evidence_manifest.json')
msa=load('msa_evidence.json'); structure=load('structure_evidence.json')
ref=case['reference_sequence']; sites={s['position']:s for s in case['evidence']['sites']}
msac={q['id']:q for q in msa['candidates']}
raw=structure['raw_coordinate_evidence']; rawsites={s['position']:s for s in raw['sites']}
assert manifest['support_count']==0 and not manifest['measurements_present']
assert case['context']['higher_is_better']
assert hashlib.sha256(ref.encode()).hexdigest()==structure['reference_sequence_sha256']
assert hashlib.sha256(ref.encode()).hexdigest()==msa['reference_sequences_sha256']['__single__']
source_checks={}
for path,spec in manifest['files'].items():
 if isinstance(spec,dict) and 'sha256' in spec:
  actual=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
  assert actual==spec['sha256'],path
  source_checks[path]=actual
# Fauchere-Pliska sidechain hydrophobicity: approximate transferable chemistry prior.
hyd=dict(zip('ACDEFGHIKLMNPQRSTVWY',[.31,1.54,-.77,-.64,1.79,0,.13,1.80,-.99,1.70,1.23,-.60,.72,-.22,-1.01,-.04,.26,1.22,2.25,.96]))

def parse(q):
 terms=[];seq=list(ref);seen=set()
 for t in q['mutation'].split(':'):
  m=re.fullmatch(r'([A-Z])(\d+)([A-Z])',t);assert m,t
  w,p,a=m.group(1),int(m.group(2)),m.group(3)
  assert ref[p-1]==w and p not in seen and a in hyd
  seen.add(p);seq[p-1]=a
  if w!=a:terms.append((w,p,a))
 assert ''.join(seq)==q['sequence']
 assert len(terms)==q['mutation_order']
 return terms

def checkpoint_effect(w,p,a,term):
 s=sites[p]['wt_structure']; r=rawsites[p].get('isolated_chain_wt_RSA')
 assert r is not None
 b=max(0,1-min(r,1))
 h=hyd[a]-hyd[w]
 e=-.18 + .75*b*h -.25*max(0,h)*(1-b)
 # Buried polar/charged replacements and secondary-structure backbone penalties.
 if w in 'AILMVFWY' and a in 'DEKRHNQST':e-=.65*b
 phi=s.get('phi_deg');psi=s.get('psi_deg')
 helix=phi is not None and psi is not None and -100<phi<-30 and -80<psi<0
 if a=='P' and w!='P':e-=1.8 if helix else .6
 if a=='G' and w!='G':e-=.8 if helix else .25
 if w=='P' and a!='P':e-=.6
 if phi is not None and phi>0 and w=='G' and a!='G':e-=1.5
 if term['sum_log_frequency_ratio'] is not None:
  e+=.30*term['sum_log_frequency_ratio']
 return e

queries=case['query'];ids=[q['id'] for q in queries]
assert len(ids)==250 and len(set(ids))==250
assert set(ids)==set(msac)
pred=[];diagnostics=[]
for q in queries:
 terms=parse(q);mt=msac[q['id']]['substitution_terms'];assert len(mt)==len(terms)
 e=0
 for (w,p,a),t in zip(terms,mt):
  assert (w,p,a)==(t['WT'],t['position'],t['ALT'])
  if sites[p]['msa']['non_gap_depth']<=0:assert t['sum_log_frequency_ratio'] is None
  e+=checkpoint_effect(w,p,a,t)
 score=float(np.tanh(e/5))
 assert math.isfinite(score) and -1<=score<=1
 pred.append({'id':q['id'],'score':round(score,7)})
 diagnostics.append({'id':q['id'],'mutation':q['mutation'],'raw_stability_prior':e,'complete_msa':msac[q['id']]['scores']['sum_log_frequency_ratio']['complete']})
summary='Label-free stability ranking combining available homolog log-frequency ratios with WT burial, sidechain chemistry and backbone compatibility; missing MSA terms use structural priors only. WT mapping and supplied source hashes verified. One global tanh scale, higher is better. No measured calibration or validated mutant energies/epistasis; substantial ranking uncertainty.'
output={'predictions':pred,'summary':summary}
(ROOT/'checkpoint_predictions.json').write_text(json.dumps(output,indent=2)+'\n')
(ROOT/'prediction_diagnostics.json').write_text(json.dumps({'source_checks':source_checks,'candidates':diagnostics},indent=2)+'\n')
print(json.dumps({'written':len(pred),'score_range':[min(x['score'] for x in pred),max(x['score'] for x in pred)],'msa_complete':sum(d['complete_msa'] for d in diagnostics),'fixed_partners':case.get('reference_chains',{}),'verified_hashes':len(source_checks)}))
