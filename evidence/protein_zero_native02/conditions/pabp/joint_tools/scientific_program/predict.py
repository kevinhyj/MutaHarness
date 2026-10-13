"""Label-free PAB1 ranking. No assay labels, fitted outcomes, or mutant coordinates.
Initial checkpoint: independent homolog marginals plus weak WT structural penalties.
"""
import json, math, hashlib, re
from pathlib import Path
import numpy as np

AA='ACDEFGHIKLMNPQRSTVWY'
HYDRO=dict(zip(AA,[1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
VOLUME=dict(zip(AA,[88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.0,116.1,140.0,227.8,193.6]))
CHARGE={a:(1 if a in 'KR' else -1 if a in 'DE' else 0) for a in AA}

def load_inputs():
 c=json.load(open('case.json')); m=json.load(open('msa_evidence.json')); s=json.load(open('structure_evidence.json')); manifest=json.load(open('evidence_manifest.json'))
 assert manifest['support_count']==0 and not manifest['measurements_present']
 assert c['context']['higher_is_better'] is True
 ref=c['reference_sequence']; sha=hashlib.sha256(ref.encode()).hexdigest()
 assert m['reference_sequences_sha256']=={'__single__':sha}
 assert s['reference_sequence_sha256']==sha
 hashes={}
 for f,meta in manifest['files'].items():
  if isinstance(meta,dict):
   h=hashlib.sha256(Path(f).read_bytes()).hexdigest(); assert h==meta['sha256']; hashes[f]=h
 ids=[q['id'] for q in c['query']]
 assert len(ids)==250 and len(set(ids))==250 and set(ids)=={'M%03d'%i for i in range(1,251)}
 assert set(ids)=={q['id'] for q in m['candidates']}=={q['id'] for q in s['candidates']}
 sites={x['position']:x for x in c['evidence']['sites']}
 for p,row in sites.items(): assert ref[p-1]==row['WT']
 raw=s['raw_coordinate_evidence']; assert raw['reference_sequence_sha256']==sha
 assert raw['mapping'][0]['method']=='exact_whole_chain_sequence'
 assert raw['mapping'][0]['mapped_observed_residues']==len(ref)
 # Single-chain target. No supplied RNA/other partner coordinates; partners are never altered.
 assert not c.get('reference_chains')
 changes={}
 for q in c['query']:
  seq=q['sequence']; assert len(seq)==len(ref)
  actual=[(p,w,a) for p,(w,a) in enumerate(zip(ref,seq),1) if w!=a]
  declared=[]
  for t in q['mutation'].split(':'):
   z=re.fullmatch(r'([A-Z])(\d+)([A-Z])',t); assert z
   w,p,a=z.groups(); p=int(p); assert ref[p-1]==w; declared.append((p,w,a))
  assert sorted(actual)==sorted(declared) and len(actual)==q['mutation_order']
  changes[q['id']]=actual
 return c,m,s,sites,changes,hashes

def structural_penalty(row,w,a):
 st=row['wt_structure']; maxasa={'A':129,'R':274,'N':195,'D':193,'C':167,'Q':225,'E':223,'G':104,'H':224,'I':197,'L':201,'K':236,'M':224,'F':240,'P':159,'S':155,'T':172,'W':285,'Y':263,'V':174}
 rsa=st.get('absolute_SASA_A2'); rsa=None if rsa is None else rsa/maxasa[w]
 conf=st.get('pLDDT'); reliability=0.5 if conf is None else min(1,max(0,(conf-50)/40))
 phi,psi=st.get('phi_deg'),st.get('psi_deg')
 pen=0.
 if rsa is not None:
  burial=max(0,1-rsa/0.4)
  if w in 'VILMFWY' and a not in 'VILMFWYAC': pen+=burial*max(0,HYDRO[w]-HYDRO[a])/3
  pen+=burial*abs(VOLUME[a]-VOLUME[w])/110
 if phi is not None and psi is not None:
  regular=(-160<phi<-40 and (-75<psi<0 or psi>80 or psi<-150))
  if a=='P' and w!='P' and regular: pen+=2.2
  if a=='G' and w!='G' and regular: pen+=0.5
  if w=='G' and a!='G' and phi>30: pen+=1.5 if a not in 'ND' else 0.7
 return reliability*pen

def main():
 c,m,s,sites,changes,hashes=load_inputs(); mix={q['id']:q for q in m['candidates']}
 diagnostics=[]; preds=[]
 for q in c['query']:
  total=0.; terms=[]
  for p,w,a in changes[q['id']]:
   row=sites[p]; f=row['msa']['frequencies']; depth=row['msa']['non_gap_depth']
   if depth>0: evo=math.log(f[a]/f[w]); observed=True
   else:
    # Prior for absent evidence, not an imputed natural homolog observation.
    evo=-0.35-0.20*abs(HYDRO[a]-HYDRO[w])-0.25*abs(CHARGE[a]-CHARGE[w]); observed=False
   pen=structural_penalty(row,w,a)
   total+=evo-0.7*pen
   terms.append({'position':p,'WT':w,'ALT':a,'msa_observed':observed,'evolution_or_fallback':evo,'WT_structure_penalty':pen})
  score=math.tanh((total+6)/8)
  preds.append({'id':q['id'],'score':round(score,7)})
  diagnostics.append({'id':q['id'],'mutation':q['mutation'],'raw_score':total,'terms':terms,'complete_msa':mix[q['id']]['scores']['sum_log_frequency_ratio']['complete']})
 out={'predictions':preds,'summary':'Unsupervised ranking from natural-homolog log-frequency ratios and weak, confidence-aware WT burial/backbone penalties; missing MSA terms use a biochemical prior, not a complete evolutionary score. Source hashes, exact WT mapping, mutation identity and all 250 IDs verified. One global monotone scale; no measured calibration, mutant relaxation, RNA-bound geometry or validated epistasis.'}
 assert len(preds)==250 and all(math.isfinite(x['score']) and -1<=x['score']<=1 for x in preds)
 Path('checkpoint_predictions.json').write_text(json.dumps(out,indent=2)+'\n')
 Path('prediction_diagnostics.json').write_text(json.dumps({'stage':'checkpoint','hashes':hashes,'candidates':diagnostics},indent=2)+'\n')
 Path('model_rationale.md').write_text('Initial checkpoint: independent homolog marginals with weak WT geometry penalties; explicit chemistry fallback for absent MSA. No phenotype supervision. Global tanh scale preserves ordering.\n')
 print('Saved usable checkpoint:',len(preds),'IDs; score range',min(x['score'] for x in preds),max(x['score'] for x in preds))
 print('Lowest:',[(x['id'],x['mutation'],round(x['raw_score'],2)) for x in sorted(diagnostics,key=lambda z:z['raw_score'])[:8]])
 print('Highest:',[(x['id'],x['mutation'],round(x['raw_score'],2)) for x in sorted(diagnostics,key=lambda z:z['raw_score'])[-8:]])
if __name__=='__main__': main()
