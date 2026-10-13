"""Unsupervised KRAS–RAF1 relative binding prior; local inputs only."""
import json, hashlib, re, math
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent
case=json.loads((BASE/'case.json').read_text())
manifest=json.loads((BASE/'evidence_manifest.json').read_text())
structure=json.loads((BASE/'structure_evidence.json').read_text())
raw=structure['raw_coordinate_evidence']
assert manifest['support_count']==0 and not manifest['measurements_present']
assert not manifest['allowed_modalities']['msa']
assert case['context']['higher_is_better']
assert hashlib.sha256((BASE/'raw_evidence/6VJJ.pdb').read_bytes()).hexdigest()==manifest['files']['raw_evidence/6VJJ.pdb']['sha256']==raw['raw_source']['sha256']
assert hashlib.sha256(case['reference_sequence'].encode()).hexdigest()==structure['reference_sequence_sha256']==raw['reference_sequence_sha256']
assert case['context']['fixed_partner_chains']=={'B':case['reference_chains']['B']}
sites={(x['chain'],x['position']):x for x in raw['sites']}
# Side-chain volume, hydropathy, and nominal side-chain charge: generic physical priors.
vol=dict(zip('ACDEFGHIKLMNPQRSTVWY',[67,86,91,109,135,48,118,124,135,124,124,96,90,114,148,73,93,105,163,141]))
hydro=dict(zip('ACDEFGHIKLMNPQRSTVWY',[1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]))
charge={a:(1 if a in 'KR' else -1 if a in 'DE' else .2 if a=='H' else 0) for a in vol}

def validate(q):
 expected={ch:list(seq) for ch,seq in case['reference_chains'].items()}
 mutations=[]
 for tok in q['mutation'].split(';'):
  m=re.fullmatch(r'([AB]):([A-Z])(\d+)([A-Z])',tok); assert m
  ch,w,p,a=m.groups(); p=int(p)
  assert ch=='A' and expected[ch][p-1]==w
  expected[ch][p-1]=a
  assert sites[(ch,p)]['WT']==w
  mutations.append((ch,w,p,a))
 assert q['sequence']==''.join(expected['A'])
 assert q['mutated_chains']=={ch:''.join(seq) for ch,seq in expected.items()}
 return mutations

# Source A has an extra initial G; canonical KRAS numbering = source position - 1.
# Refinement: WT backbone and actual contacts constrain position-specific risks.
# No mutant coordinates or pair energies are inferred.
import structure_tools as tools
residues=tools._read_pdb(BASE/'raw_evidence/6VJJ.pdb',include_zero_occupancy=True)
residues,verified_mapping=tools._map_raw_residues(tools._prepare(case),residues,None)
assert verified_mapping==raw['mapping']
atoms={(r['chain'],r['position']):r['atoms'] for r in residues if r['protein']}

def coord(ch,p,name):
 return np.array(atoms[(ch,p)][name]['coord'],dtype=float)

def torsion(a,b,c,d):
 b0=a-b; b1=c-b; b2=d-c; b1/=np.linalg.norm(b1)
 v=b0-np.dot(b0,b1)*b1; w=b2-np.dot(b2,b1)*b1
 return float(np.degrees(np.arctan2(np.dot(np.cross(b1,v),w),np.dot(v,w))))

features={}
for (ch,p),s in sites.items():
 phi=torsion(coord(ch,p-1,'C'),coord(ch,p,'N'),coord(ch,p,'CA'),coord(ch,p,'C')) if (ch,p-1) in atoms else None
 psi=torsion(coord(ch,p,'N'),coord(ch,p,'CA'),coord(ch,p,'C'),coord(ch,p+1,'N')) if (ch,p+1) in atoms else None
 # Nonlocal backbone N/O proximity only; no assertion of a validated hydrogen bond.
 nh_contacts=sum(np.linalg.norm(coord(ch,p,'N')-coord(c,j,'O'))<3.35 for (c,j) in atoms if (c!=ch or abs(p-j)>2) and 'O' in atoms[(c,j)])
 salt=0.; polar=0.; hydrophobic=0.
 for contact in s['cross_chain_heavy_atom_contacts']:
  c=contact['neighbor']['chain']; j=contact['neighbor']['position']; partner_w=contact['neighbor']['WT']
  for name,at in atoms[(ch,p)].items():
   if name in ['N','CA','C','O','OXT'] or at['element']=='H': continue
   for bn,bt in atoms[(c,j)].items():
    if bn in ['N','CA','C','O','OXT'] or bt['element']=='H': continue
    distance=np.linalg.norm(np.array(at['coord'])-np.array(bt['coord']))
    if distance<4.1 and charge[s['WT']]*charge[partner_w]<-.5 and name[0] in 'NO' and bn[0] in 'NO':
     salt=max(salt,float(np.clip((4.3-distance)/1.3,0,1)))
    if distance<3.6 and name[0] in 'NO' and bn[0] in 'NO': polar=max(polar,float(np.clip((3.8-distance)/1.,0,1)))
    if distance<4.5 and name[0]=='C' and bn[0]=='C': hydrophobic=max(hydrophobic,float(np.clip((4.8-distance)/1.3,0,1)))
 features[(ch,p)]={'phi':phi,'psi':psi,'nonlocal_backbone_N_O_contacts':int(nh_contacts),'partner_salt_proximity':salt,'partner_polar_proximity':polar,'partner_carbon_proximity':hydrophobic}

# GTP occupancy priors, with conservative magnitude because cellular assay context is incomplete.
activation_prior={
 (12,'C'):.72,(12,'D'):.96,(12,'V'):1.08,(12,'M'):.65,
 (61,'L'):1.12,(61,'I'):1.05,(61,'V'):1.,(61,'W'):.85,(61,'N'):.50,(61,'E'):.25,
 # V14I-like substitutions and noncanonical activating sites are not extrapolated indiscriminately.
}

def site_effect(ch,w,p,a):
 if w==a: return dict(folding=0.,backbone=0.,interface=0.,nucleotide=0.,activation=0.)
 s=sites[(ch,p)]; f=features[(ch,p)]; n=p-1
 # This dataset has complete observed WT heavy atoms. Missing values would be absent evidence.
 rsa=s['isolated_chain_wt_RSA']
 assert rsa is not None and s['backbone_complete'] and s['sidechain_complete']
 burial=float(np.clip(1-rsa/.65,0,1))
 dv=vol[a]-vol[w]
 polarity_loss=max(0.,hydro[w]-hydro[a])
 folding=.05 + .045*abs(hydro[w]-hydro[a]) + .05*abs(charge[w]-charge[a])
 folding+=burial*(.010*max(dv,0)+.006*max(-dv,0)+.12*polarity_loss)
 if w in 'ILVMFWYAC' and a in 'DEKR': folding+=.70*burial
 if w in 'FWY' and a not in 'FWY': folding+=.20*burial
 # Conservative changes have smaller cavity/packing effects.
 if w in 'ILVM' and a in 'ILVM': folding*=.6
 if {w,a}<={'F','Y'} or {w,a}<={'D','E'} or {w,a}<={'K','R'}: folding*=.45
 phi=f['phi']; psi=f['psi']; backbone=0.
 helical=phi is not None and -100<phi<-30 and psi is not None and -85<psi<15
 if a=='P' and w!='P':
  incompatible=phi is not None and (phi>0 or phi<-100)
  backbone=.35 + (.9 if incompatible else 0) + (.60 if f['nonlocal_backbone_N_O_contacts']>0 else 0)
  if helical and f['nonlocal_backbone_N_O_contacts']>0: backbone+=.25
 if a=='G' and w!='G': backbone=.2 + (.4 if helical else .1)*burial
 if w=='G' and a!='G':
  backbone=.15 + (.65 if phi is not None and phi>0 else 0)
  if a in 'VIT' and phi is not None and phi>0: backbone+=.25
 if w=='P' and a!='P': backbone=.18+.12*burial
 # Side-chain-dependent binding loss from fixed B contacts; backbone-only contacts
 # do not receive side-chain salt-bridge penalties.
 interface=0.
 salt=f['partner_salt_proximity']
 if salt:
  if charge[a]*charge[w]>0: interface+=.25*salt if a!=w else 0.
  elif charge[a]*charge[w]<0: interface+=1.65*salt
  else: interface+=1.05*salt
 if f['partner_polar_proximity'] and w in 'STNQH' and a not in 'STNQHDEKR': interface+=.22*f['partner_polar_proximity']
 if w in 'ILVMFWY' and f['partner_carbon_proximity']:
  contact_weight=min(1.,s['observed_atom_SASA_buried_by_context_A2']/40.)
  if a not in 'ILVMFWY': interface+=.45*contact_weight
  if w in 'FWY' and a not in 'FWY': interface+=.35*contact_weight
 # Known RAF-recognition residues and nucleotide motifs: function, separate from folding.
 nucleotide=0.
 if n==17: nucleotide+=2.65 if a=='N' else 1.9
 if n==35: nucleotide+=1.2 if a=='S' else 1.8
 if n==57: nucleotide+=1.6 if a!='E' else .35
 if n==16: nucleotide+=1.65 if a not in 'KR' else .3
 if n==15: nucleotide+=1.1
 if n in [10,11,13,14]: nucleotide+=.45
 if n in [59,60]: nucleotide+=.45 if a in 'FWYKR' else .2
 if n==116: nucleotide+=.9 if a not in 'NQ' else .15
 if n==117: nucleotide+=.65 if a!='R' else .15
 if n==119: nucleotide+=1.4 if a not in 'DE' else .2
 if n==145: nucleotide+=.25
 if n==146 and a in 'FWYKR': nucleotide+=.55
 # Preserve acidic identity at D38; chemistry is consistent with WT partner contacts.
 if n==38 and a not in 'DE': interface+=.35
 if n==37 and a not in 'DE': interface+=.15
 if n==40 and a not in 'FWY': interface+=.45
 # Switch-II flexibility is sensitive, but A66 is a helix-start position with no
 # nonlocal N/O contact: it must not receive a generic helix-interior proline loss.
 if n in [63,64,66] and a!=w: nucleotide+=.10
 activation=activation_prior.get((n,a),0.)
 if n==12 and activation:
  folding=min(folding,.12 if a in 'CD' else .20); backbone=.05
 if n==61 and activation:
  folding*=.45
 return {k:float(v) for k,v in dict(folding=folding,backbone=backbone,interface=interface,nucleotide=nucleotide,activation=activation).items()}

if __name__=='__main__':
 predictions=[]; details=[]
 for q in case['query']:
  muts=validate(q); effects=[site_effect(*m) for m in muts]
  loss=sum(sum(e[k] for k in ['folding','backbone','interface','nucleotide']) for e in effects)
  bonuses=sorted([e['activation'] for e in effects],reverse=True)
  # G12 + Q61 both favor the same finite GTP-bound pool: do not add two full gains.
  activation=bonuses[0]+.15*sum(bonuses[1:]) if bonuses else 0.
  latent=.10 + activation-loss
  score=round(float(np.tanh(latent/2.7)),6)
  predictions.append({'id':q['id'],'score':score})
  details.append({'id':q['id'],'mutation':q['mutation'],'site_effects':effects,'loss':loss,'activation':activation,'latent':latent,'score':score})
 assert len(predictions)==250 and len({p['id'] for p in predictions})==250
 assert {p['id'] for p in predictions}=={f'M{i:03}' for i in range(1,251)}
 assert all(math.isfinite(p['score']) and -1<=p['score']<=1 for p in predictions)
 summary='Zero-shot KRAS–RAF1 binding ranking combines mutation chemistry, verified WT burial/backbone and fixed RAF1 contact chemistry with KRAS nucleotide-state and effector-site priors. Source/sequence hashes and complete A/B residue mapping checked; no MSA or assay outcomes. One global tanh scale; static modeled WT coordinates, uncertain state effects and unvalidated epistasis limit confidence.'
 output={'predictions':predictions,'summary':summary}
 for fn in ['checkpoint_predictions.json','final_predictions.json']:
  (BASE/fn).write_text(json.dumps(output,indent=2)+'\n')
 (BASE/'prediction_details.json').write_text(json.dumps(details,indent=2)+'\n')
 (BASE/'model_features.json').write_text(json.dumps([dict(chain=c,position=p,**f) for (c,p),f in features.items()],indent=2)+'\n')
 print('Final coverage',len(predictions),'range',min(x['score'] for x in predictions),max(x['score'] for x in predictions))
 print('Top 12',[(d['id'],d['mutation'],d['score']) for d in sorted(details,key=lambda d:d['score'],reverse=True)[:12]])
 print('Bottom 8',[(d['id'],d['mutation'],d['score']) for d in sorted(details,key=lambda d:d['score'])[:8]])
