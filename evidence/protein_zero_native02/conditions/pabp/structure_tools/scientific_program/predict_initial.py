"""Unsupervised PAB1 ranking from supplied WT evidence; no assay outcomes."""
import json, re, math, hashlib
from pathlib import Path
import numpy as np

HYDRO=dict(zip('ACDEFGHIKLMNPQRSTVWY',[1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
VOL=dict(zip('ACDEFGHIKLMNPQRSTVWY',[88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.0,116.1,140.0,227.8,193.6]))
CHARGE={a:0 for a in HYDRO};CHARGE.update(K=1,R=1,D=-1,E=-1,H=.2)
AROM=set('FWY')

def load_inputs():
 c=json.load(open('case.json')); e=json.load(open('structure_evidence.json')); m=json.load(open('evidence_manifest.json'))
 assert m['support_count']==0 and not m['measurements_present'] and c['context']['higher_is_better']
 for p,v in m['files'].items():
  if isinstance(v,dict) and 'sha256' in v:
   assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==v['sha256'],p
 assert hashlib.sha256(c['reference_sequence'].encode()).hexdigest()==e['reference_sequence_sha256']
 sites={s['position']:s for s in e['raw_coordinate_evidence']['sites'] if s['chain'] is None}
 desc={s['position']:s['wt_structure'] for s in c['evidence']['sites'] if s['chain'] is None}
 assert len({q['id'] for q in c['query']})==250
 for q in c['query']:
  seq=list(c['reference_sequence']);seen=set()
  for token in q['mutation'].split(':'):
   w,p,a=re.fullmatch(r'([A-Z])(\d+)([A-Z])',token).groups();p=int(p)
   assert p not in seen and seq[p-1]==w and sites[p]['WT']==w
   seen.add(p);seq[p-1]=a
  assert ''.join(seq)==q['sequence'] and len(seen)==q['mutation_order']
 return c,e,sites,desc

def site_cost(w,p,a,sites,desc):
 if w==a:return 0.0,{}
 s=sites[p];d=desc[p];rsa=s['isolated_chain_wt_RSA']
 # Exponent smoothly distinguishes a buried core from an exposed site. Missing
 # geometry is absent evidence, never zero burial; all queried sites are present.
 assert rsa is not None
 burial=max(0,1-min(1,rsa)/.65)**1.5
 chemical=.18+.14*abs(HYDRO[w]-HYDRO[a])+.35*abs(CHARGE[w]-CHARGE[a])+.35*(abs(VOL[w]-VOL[a])/100)
 cost=.18*chemical
 core=burial*(.30*abs(HYDRO[w]-HYDRO[a])+.85*abs(VOL[w]-VOL[a])/100)
 if w in 'ILVMFAWY' and a in 'DENQKRH':core+=1.2*burial
 cost+=core
 phi,psi=d['phi_deg'],d['psi_deg']
 backbone=0
 if a=='P' and w!='P':
  backbone=1.15 if (phi is not None and phi < -90) else .30
  if p in range(140,145) or p in range(178,185):backbone+=1.3
 if a=='G' and w!='G':backbone+=.45*burial +(.4 if p in range(140,146) or p in range(178,185) else 0)
 if w=='G' and phi is not None and phi>0:
  backbone+=.65 if a in 'ASND' else 1.45
 if w=='P' and a!='P':backbone+=.65
 cost+=backbone
 motif=0
 if p in (129,168,170) and w in AROM:
  motif=0 if a in AROM else (1.15 if a in 'ILVM' else 1.8)
 if p in (131,166) and a not in 'KR':motif+=.45 if a not in 'DE' else .8
 if p==169 and a!='G':motif+=.5
 cost+=motif
 return cost,dict(chemical=chemical,core=core,backbone=backbone,motif=motif,rsa=rsa)

def main():
 c,e,sites,desc=load_inputs();pred=[];details=[]
 for q in c['query']:
  vals=[]; ds=[]
  for t in q['mutation'].split(':'):
   w,p,a=re.fullmatch(r'([A-Z])(\d+)([A-Z])',t).groups();p=int(p)
   value,parts=site_cost(w,p,a,sites,desc);vals.append(value);ds.append(dict(mutation=t,cost=value,**parts))
  cost=sum(vals)
  score=float(2*math.exp(-cost/.95)-1)
  assert math.isfinite(score) and -1<=score<=1
  pred.append(dict(id=q['id'],score=round(score,6)));details.append(dict(id=q['id'],mutation=q['mutation'],cost=cost,sites=ds))
 result=dict(predictions=pred,summary='Zero-shot ranking from hash-verified, exactly mapped WT structure: burial-dependent chemistry/packing, backbone compatibility, and PAB1 RRM RNA-binding motif priors. No measured supports or MSA; no partner coordinates, mutant relaxation or validated epistasis. Scores use one global monotone scale and are uncertain relative-performance predictions, not calibrated assay values.')
 assert {x['id'] for x in pred}=={f'M{i:03}' for i in range(1,251)}
 Path('checkpoint_predictions.json').write_text(json.dumps(result,indent=2)+'\n')
 Path('prediction_details.json').write_text(json.dumps(details,indent=2)+'\n')
 print('Saved checkpoint_predictions.json: 250 finite scores; range',min(x['score'] for x in pred),max(x['score'] for x in pred))
if __name__=='__main__':main()
