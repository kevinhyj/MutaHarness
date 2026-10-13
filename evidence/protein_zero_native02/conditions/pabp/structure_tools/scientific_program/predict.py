"""Unsupervised PAB1 ranking from supplied WT evidence; no assay outcomes."""
import json, re, math, hashlib
from pathlib import Path
import numpy as np
from amino_acid_priors import BLOSUM, similarity_penalty
from repeat_evidence import build as build_repeats
from backbone_evidence import build as build_backbone

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

def site_cost(w,p,a,sites,desc,repeats,backbone_evidence):
 if w==a:return 0.0,{}
 s=sites[p];d=desc[p];rsa=s['isolated_chain_wt_RSA']
 assert rsa is not None and s['backbone_complete'] and s['sidechain_complete']
 burial=max(0,1-min(1,rsa)/.65)**1.5
 # Confidence changes reliance on geometry, not predicted assay performance.
 reliability=min(1,max(0,(s['source_asserted_AF2_pLDDT']-50)/35))
 burial*=reliability
 generic=.08+.26*similarity_penalty(w,a)
 # Directional polarity changes: desolvating a new charge is distinct from
 # losing an existing buried polar contact. Packing is a modest WT-volume
 # prior, not a mutant clash or energy calculation.
 dv=(VOL[a]-VOL[w])/70
 packing=burial*(1.05*max(0,dv)**2+.70*max(0,-dv))
 nonpolar=set('ACFILMVWY')
 polarity=0.0
 if w in nonpolar and a not in nonpolar and a not in 'GP':
  polarity+=burial*(.25+.22*max(0,HYDRO[w]-HYDRO[a])+.45*abs(CHARGE[a]))
 elif w not in nonpolar and w not in 'GP' and a in nonpolar:
  polarity+=.55*burial
 if w not in nonpolar and w not in 'GP' and a not in nonpolar:
  polarity+=.35*burial*abs(CHARGE[w]-CHARGE[a])
 if w in AROM and a not in AROM:packing+=.40*burial
 # Intra-protein repeat compatibility is weak sequence evidence from exactly
 # two WT repeats, not independent homologs, conservation depth or an MSA.
 peers=repeats['sites'].get(str(p),[])
 peer_delta=np.mean([BLOSUM[w,x['WT']]-BLOSUM[a,x['WT']] for x in peers]) if peers else None
 repeat=0 if peer_delta is None else .085*float(peer_delta)
 repeat=max(-.20,min(1.0,repeat))
 # Restrict positive-phi penalties using the observed repeat chemistry. A
 # glycine here need not be glycine-essential if non-G repeats have the same
 # fold and closely corresponding WT coordinates.
 gly_conserved=sum(x['WT']=='G' for x in peers)/len(peers) if peers else None
 gly_factor=1.0 if gly_conserved is None else .55+.45*gly_conserved
 phi,psi=d['phi_deg'],d['psi_deg'];backbone=0.0
 bb=backbone_evidence['sites'][str(p)]
 if a=='P' and w!='P':
  backbone=.25
  if phi is not None and phi < -90:backbone+=.7
  if bb['nonlocal_backbone_N_O_distance_contacts']:backbone+=.7
  if bb['has_i_i_plus4_backbone_contact']:backbone+=.3
 if a=='G' and w!='G':
  backbone+=.30*burial
  if bb['has_i_i_plus4_backbone_contact']:backbone+=.55
 if phi is not None and phi>0 and a!='G':
  if w=='G':
   backbone+=(.65 if a in 'ASND' else 1.45)*gly_factor
   if phi>110:backbone+=.55
  elif a not in 'NSD':backbone+=.35 if a in 'AT' else .65
 if w=='P' and a!='P':backbone+=.30
 # RNP2/RNP1 aromatic residues contact RNA; no partner is mutated and no RNA
 # coordinates are supplied. The motif prior does not claim affinity values.
 motif=0.0
 if p in (129,168,170) and w in AROM:
  motif=0 if a in AROM else (.85 if a in 'ILVM' else 1.25)
 if p in (131,166) and a not in 'KR':motif+=.20 if a not in 'DE' else .45
 if p==169 and a!='G':motif+=.25
 cost=max(.025,generic+packing+polarity+repeat+backbone+motif)
 return cost,dict(generic=generic,packing=packing,polarity=polarity,repeat=repeat,backbone=backbone,motif=motif,rsa=rsa,geometry_reliability=reliability)

def main():
 c,e,sites,desc=load_inputs();pred=[];details=[]
 repeats=build_repeats(c,e)
 backbone=build_backbone(c)
 Path('backbone_contact_evidence.json').write_text(json.dumps(backbone,indent=2)+'\n')
 Path('internal_repeat_evidence.json').write_text(json.dumps(repeats,indent=2)+'\n')
 for q in c['query']:
  vals=[]; ds=[]
  for t in q['mutation'].split(':'):
   w,p,a=re.fullmatch(r'([A-Z])(\d+)([A-Z])',t).groups();p=int(p)
   value,parts=site_cost(w,p,a,sites,desc,repeats,backbone);vals.append(value);ds.append(dict(mutation=t,cost=value,**parts))
  cost=sum(vals)
  score=float(2*math.exp(-cost/1.5)-1)
  assert math.isfinite(score) and -1<=score<=1
  pred.append(dict(id=q['id'],score=round(score,6)));details.append(dict(id=q['id'],mutation=q['mutation'],cost=cost,sites=ds))
 result=dict(predictions=pred,summary='Unsupervised PAB1 ranking combining hash-verified WT burial/backbone geometry, physicochemical and BLOSUM62 priors, RNA-binding motifs, and weak compatibility with two domains in the same reference protein. All mutated residues map exactly and have complete coordinates. No MSA or measured outcomes are available. Double-mutant costs share one global score scale. Growth effects, RNA affinity, mutant relaxation and epistasis remain uncertain; scores are relative and uncalibrated.')
 assert {x['id'] for x in pred}=={f'M{i:03}' for i in range(1,251)}
 Path('checkpoint_predictions.json').write_text(json.dumps(result,indent=2)+'\n')
 Path('final_predictions.json').write_text(json.dumps(result,indent=2)+'\n')
 Path('prediction_details.json').write_text(json.dumps(details,indent=2)+'\n')
 print('Saved checkpoint_predictions.json: 250 finite scores; range',min(x['score'] for x in pred),max(x['score'] for x in pred))
if __name__=='__main__':main()
