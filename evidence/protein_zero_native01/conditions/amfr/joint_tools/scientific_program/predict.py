"""Label-free stability prior; no assay observations or fitted parameters.
Checkpoint version: additive burial, helix and weak natural profile compatibility.
"""
import json, math, re, hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
# Published-scale physicochemical priors (hydropathy and helix free-energy costs).
KD=dict(zip('ACDEFGHIKLMNPQRSTVWY',[1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]))
HELIX=dict(zip('ACDEFGHIKLMNPQRSTVWY',[0,.68,.69,.40,.54,1,.61,.41,.26,.21,.24,.65,3.16,.39,.21,.50,.66,.61,.49,.53]))
VOL=dict(zip('ACDEFGHIKLMNPQRSTVWY',[88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89,116.1,140,227.8,193.6]))

def load_validate():
 c=json.load(open(ROOT/'case.json')); m=json.load(open(ROOT/'msa_evidence.json')); s=json.load(open(ROOT/'structure_evidence.json'));manifest=json.load(open(ROOT/'evidence_manifest.json'))
 assert not manifest['measurements_present'] and manifest['support_count']==0
 ref=c['reference_sequence']; rh=hashlib.sha256(ref.encode()).hexdigest()
 assert m['reference_sequences_sha256']['__single__']==rh==s['reference_sequence_sha256']
 for file,meta in manifest['files'].items():
  if isinstance(meta,dict) and 'sha256' in meta:
   assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==meta['sha256'],file
 assert c['context']['higher_is_better']
 assert len(c['query'])==250 and {q['id'] for q in c['query']}=={f'M{i:03d}' for i in range(1,251)}
 for q in c['query']:
  muts=[]; seq=list(ref)
  for token in q['mutation'].split(':'):
   match=re.fullmatch(r'([A-Z])(\d+)([A-Z])',token); assert match
   w,p,a=match.groups();p=int(p); assert ref[p-1]==w and w!=a
   seq[p-1]=a;muts.append((p,w,a))
  assert ''.join(seq)==q['sequence'] and len(muts)==q['mutation_order']
  q['parsed']=muts
 # This stability case has no supplied partner sequences. No biological assembly inferred.
 assert 'reference_chains' not in c
 return c,m,s,manifest

def checkpoint():
 c,m,s,manifest=load_validate(); sites={x['position']:x for x in c['evidence']['sites']};raw={x['position']:x for x in s['raw_coordinate_evidence']['sites']};mi={x['id']:x for x in m['candidates']}
 preds=[];diagnostics=[]
 for q in c['query']:
  physics=0.;evo=0.;covered=0
  for p,w,a in q['parsed']:
   st=sites[p]['wt_structure']; r=raw[p]['isolated_chain_wt_RSA'];bur=max(0.,1-min(1.,r)/.7)
   physics+=.22*bur*(KD[a]-KD[w])
   physics-=.00012*bur*(VOL[a]-VOL[w])**2
   if st.get('phi_deg') is not None and st.get('psi_deg') is not None and -100<st['phi_deg']<-30 and -80<st['psi_deg']<0:
    physics-=.8*(HELIX[a]-HELIX[w])
   if w=='G' and (st.get('phi_deg') or 0)>0:physics-=2.
   term=next(t for t in mi[q['id']]['substitution_terms'] if t['position']==p)
   if term['sum_log_frequency_ratio'] is not None and term['non_gap_depth']>0:
    evo+=max(-5.,min(3.,term['sum_log_frequency_ratio']));covered+=1
  total=physics+.14*evo-.15*len(q['parsed'])
  preds.append({'id':q['id'],'score':round(math.tanh(total/4.),6)})
  diagnostics.append({'id':q['id'],'mutation':q['mutation'],'physics':physics,'observed_evolutionary_terms':covered,'required_terms':len(q['parsed']),'evolutionary_partial_contribution':evo,'raw_score':total})
 result={'predictions':preds,'summary':'Unsupervised stability prior combines WT burial, volume compatibility and backbone/helix propensity with weak observed homolog log-frequency terms. Source hashes, WT residue mapping and mutation sequences verified; zero-depth MSA terms omitted and coverage tracked. One global tanh scale, higher is better. No measured calibration, mutant relaxation or validated epistasis; rankings are uncertain.'}
 json.dump(result,open(ROOT/'checkpoint_predictions.json','w'),indent=2);json.dump(diagnostics,open(ROOT/'checkpoint_diagnostics.json','w'),indent=2)
 print('Checkpoint saved:',len(preds),'finite scores; range',min(x['score'] for x in preds),max(x['score'] for x in preds))

if __name__=='__main__':checkpoint()
