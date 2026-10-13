"""Unsupervised stability ranking from supplied public WT/MSA evidence.
All coefficients are fixed physical heuristics, not learned from assay outcomes.
WT contacts describe chemical opportunities, never relaxed mutant interactions.
"""
import json, math, hashlib
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from predict import load_validate, HELIX, VOL
ROOT=Path(__file__).resolve().parent
# Fauchere-Pliska water/octanol sidechain hydrophobicity: soluble-core prior.
TRANSFER=dict(zip('ACDEFGHIKLMNPQRSTVWY',[.31,1.54,-.77,-.64,1.79,0,.13,1.8,-.99,1.7,1.23,-.60,.72,-.22,-1.01,-.04,.26,1.22,2.25,.96]))
CHARGE={a: (-1 if a in 'DE' else 1 if a in 'KR' else 0) for a in TRANSFER}
DONORS={'R':['NE','NH1','NH2'],'N':['ND2'],'Q':['NE2'],'K':['NZ'],'S':['OG'],'T':['OG1'],'Y':['OH'],'H':['ND1','NE2']}
ACCEPTORS={'D':['OD1','OD2'],'E':['OE1','OE2'],'N':['OD1'],'Q':['OE1'],'S':['OG'],'T':['OG1'],'Y':['OH'],'H':['ND1','NE2']}
REACH=dict(zip('ACDEFGHIKLMNPQRSTVWY',[1,2.0,2.6,3.8,3.9,0,3.5,2.8,5,3.5,4,2.6,2.0,3.8,5.4,1.9,1.9,2.4,4.5,5.0]))
AA3={'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}
GROUPS=['AFILMVWY','GC','ST','NQH','DE','KR','P']
GROUP={a:i for i,aa in enumerate(GROUPS) for a in aa}

def parse_atoms(ref):
 atoms=defaultdict(dict);names={};chains=set()
 for line in open(ROOT/'raw_evidence/AMFR_HUMAN.pdb'):
  if line.startswith('ENDMDL'):break
  if line.startswith('ATOM'):
   p=int(line[22:26]);chain=line[21];a=line[12:16].strip();el=line[76:78].strip()
   assert chain=='A' and not line[16].strip() and float(line[54:60])>0
   chains.add(chain);names[p]=AA3[line[17:20].strip()]
   atoms[p][a]=(el,np.array([float(line[30:38]),float(line[38:46]),float(line[46:54])]))
 assert ''.join(names[p] for p in sorted(names))==ref and sorted(names)==list(range(1,len(ref)+1))
 assert chains=={'A'}
 return atoms

def torsion(a,b,c,d):
 b0=a-b;b1=c-b;b2=d-c;b1=b1/np.linalg.norm(b1)
 v=b0-np.dot(b0,b1)*b1;w=b2-np.dot(b2,b1)*b1
 return float(np.degrees(np.arctan2(np.dot(np.cross(b1,v),w),np.dot(v,w))))

def role_retention(w,a,atom,role):
 if w==a:return 1.
 if atom in ('N','O','OXT'):return 0. if atom=='N' and a=='P' else 1.
 available=DONORS if role=='donor' else ACCEPTORS
 if not available.get(a):return 0.
 # Preserve close chemical isosteres, but changes in length/branching are uncertain.
 ret=.80*math.exp(-abs(REACH[a]-REACH[w])/2.5)
 if {w,a}<={'S','T'}:ret=.90
 if {w,a}<={'D','N'}:
  ret=.85 if atom=='OD1' else .30
 if {w,a}<={'E','Q'}:ret=.85 if atom=='OE1' else .30
 if {w,a}<={'N','Q'}:ret=.65
 if a=='H':ret=min(ret,.70)
 return ret

def hbond_edges(atoms,ref,bur):
 # Directional explicit-H WT candidates. Select one atom pair per residue/role edge
 # to avoid counting bifurcated carboxylate contacts twice.
 candidates={}
 for p,w in enumerate(ref,1):
  for dn in DONORS.get(w,[])+(['N'] if w!='P' else []):
   if dn not in atoms[p]:continue
   d=atoms[p][dn][1];hs=[x for e,x in atoms[p].values() if e=='H' and np.linalg.norm(x-d)<1.3]
   for q,a in enumerate(ref,1):
    if p==q:continue
    for an in ACCEPTORS.get(a,[])+['O','OXT']:
     if an not in atoms[q] or dn=='N' and an in ('O','OXT'):continue
     ac=atoms[q][an][1];dist=float(np.linalg.norm(d-ac))
     if dist>3.8:continue
     angles=[float(np.degrees(np.arccos(np.clip(np.dot(d-h,ac-h)/np.linalg.norm(d-h)/np.linalg.norm(ac-h),-1,1)))) for h in hs]
     angle=max(angles or [0])
     if angle<=110:continue
     quality=min(1.,max(0.,(angle-100)/50))*math.exp(-max(0.,dist-3.25)/.5)
     strength=(.60+.50*(bur[p]+bur[q])/2)*quality
     key=(p,q,dn in ('N',),an in ('O','OXT'))
     if key not in candidates or strength>candidates[key]['strength']:
      candidates[key]={'donor_pos':p,'donor_atom':dn,'acceptor_pos':q,'acceptor_atom':an,'distance_A':dist,'angle_deg':angle,'strength':strength}
 return list(candidates.values())

def alignment_prior(c):
 # Same 2048 lexically sorted unique rows and effective weights as supplied profile.
 # Reconstruction script separately verifies marginal equality and Neff.
 z=np.load(ROOT/'alignment_reconstructed.npz');x=z['x'];w=z['weights']
 assert abs(w.sum()-631.201790072901)<1e-8
 ref=c['reference_sequence']
 rows={r['position']:r for r in c['evidence']['sites']}
 for p in range(6,47):
  counts=Counter()
  for a,ww in zip(x[:,p-6],w):
   if a in TRANSFER:counts[a]+=float(ww)
  dep=sum(counts.values());ms=rows[p]['msa']
  assert abs(dep-ms['non_gap_depth'])<1e-8
  assert max(abs((counts[a]+.05)/(dep+1)-ms['frequencies'][a]) for a in TRANSFER)<1e-10
 # Small chemistry-group joint prior restricted to observed buried-contact pairs.
 # 7x7 tables with 50 effective observations of independent-marginal shrinkage.
 # This is natural covariation, not experimentally established epistasis.
 pairs={}
 for p,q in [(28,38),(28,41),(32,38),(32,41)]:
  joint=np.zeros((7,7))
  for a,b,ww in zip(x[:,p-6],x[:,q-6],w):
   if a in GROUP and b in GROUP:joint[GROUP[a],GROUP[b]]+=ww
  depth=float(joint.sum());f=joint/depth;ind=f.sum(1)[:,None]*f.sum(0)[None,:]
  shrunk=(joint+50*ind+1/49)/(depth+51)
  marginal=shrunk.sum(1)[:,None]*shrunk.sum(0)[None,:]
  enrichment=np.log(shrunk/marginal)
  pairs[p,q]={'enrichment':enrichment,'coverage_depth':depth}
 return pairs

def predict():
 c,m,s,manifest=load_validate();ref=c['reference_sequence'];atoms=parse_atoms(ref)
 raw={r['position']:r for r in s['raw_coordinate_evidence']['sites']};sites={r['position']:r for r in c['evidence']['sites']};mi={r['id']:r for r in m['candidates']}
 assert s['raw_coordinate_evidence']['raw_source']['sha256']==manifest['files']['raw_evidence/AMFR_HUMAN.pdb']['sha256']
 assert s['raw_coordinate_evidence']['mapping'][0]['mapped_observed_residues']==len(ref)
 for p,r in raw.items():
  assert r['WT']==ref[p-1] and r['coordinate_present'] and r['backbone_complete'] and r['sidechain_complete']
  assert np.linalg.norm(atoms[p]['CA'][1]-np.array(r['CA_coordinate_A']))<1e-8
 rsa={p:raw[p]['isolated_chain_wt_RSA'] for p in raw}
 bur={p:max(0.,1-min(1.,rsa[p])/.85) for p in rsa}
 phi={};psi={}
 for p in atoms:
  if p>1:phi[p]=torsion(atoms[p-1]['C'][1],atoms[p]['N'][1],atoms[p]['CA'][1],atoms[p]['C'][1])
  if p<len(ref):psi[p]=torsion(atoms[p]['N'][1],atoms[p]['CA'][1],atoms[p]['C'][1],atoms[p+1]['N'][1])
  for name,computed in [('phi_deg',phi.get(p)),('psi_deg',psi.get(p))]:
   supplied=sites[p]['wt_structure'].get(name)
   if supplied is not None:assert abs(supplied-computed)<.02
 alpha_like={p:p in phi and p in psi and -100<phi[p]<-30 and -80<psi[p]<0 for p in atoms}
 # Require a contiguous alpha-like stretch: isolated P18/Q19 and R46 are turns,
 # so removing native loop proline must not get a helix-propensity windfall.
 helical={p:False for p in atoms};run=[]
 for p in list(atoms)+[len(ref)+1]:
  if alpha_like.get(p,False):run.append(p)
  else:
   if len(run)>=4:
    for r in run:helical[r]=True
   run=[]
 edges=hbond_edges(atoms,ref,bur);pairs=alignment_prior(c)
 # Actual nonlocal heavy sidechain contacts give a bounded packing constraint.
 constraint={};contact_pairs={}
 for p in atoms:
  neighbors=set()
  for q in atoms:
   if abs(p-q)<=2:continue
   distances=[np.linalg.norm(x-y) for a,(e,x) in atoms[p].items() for b,(f,y) in atoms[q].items() if e!='H' and f!='H' and a not in ('N','CA','C','O','OXT') and b not in ('N','CA','C','O','OXT')]
   if distances and min(distances)<4.5:
    neighbors.add(q);contact_pairs[tuple(sorted((p,q)))]=float(min(distances))
  constraint[p]=.55+.45*np.mean([bur[q] for q in neighbors]) if neighbors else .55
 diagnostics=[];preds=[]
 for q in c['query']:
  sequence=q['sequence'];mutpos={p for p,_,_ in q['parsed']};f=defaultdict(float)
  for p,w,a in q['parsed']:
   b=bur[p];delta=VOL[a]-VOL[w]
   f['nonpolar_transfer']+=b*(TRANSFER[a]-TRANSFER[w])-.12*min(1,rsa[p])*max(0,TRANSFER[a]-TRANSFER[w])
   f['charge_desolvation']-=1.20*b*(abs(CHARGE[a])-abs(CHARGE[w]))
   if delta<0:f['cavity']-=.018*b*(-delta)
   else:f['overpacking']-=b*constraint[p]*(.006*delta+.00020*max(0,delta-20)**2)
   if helical[p]:
    # Last alpha-like residue before a turn can accommodate a helix stop more readily.
    hw=.45 if not helical.get(p+1,False) else .80
    f['backbone_propensity']-=hw*(HELIX[a]-HELIX[w])
   elif w=='P':f['backbone_propensity']-=.85
   elif a=='P':f['backbone_propensity']-=.8
   if (phi.get(p) or 0)>0:
    if w=='G':f['backbone_propensity']-=2.5 if a not in 'DNST' else 1.5
    elif a not in 'GDNSTC':f['backbone_propensity']-=.65
   # Weight observed evolutionary terms conservatively; zero depth is absent evidence.
   t=next(t for t in mi[q['id']]['substitution_terms'] if t['position']==p)
   if t['sum_log_frequency_ratio'] is not None and t['non_gap_depth']>0:
    reliability=t['non_gap_depth']/(t['non_gap_depth']+50)
    f['natural_marginal']+=.18*reliability*np.clip(t['sum_log_frequency_ratio'],-7,3)
  for edge in edges:
   p=edge['donor_pos'];r=edge['acceptor_pos']
   if not (p in mutpos or r in mutpos):continue
   retain_d=role_retention(ref[p-1],sequence[p-1],edge['donor_atom'],'donor')
   retain_a=role_retention(ref[r-1],sequence[r-1],edge['acceptor_atom'],'acceptor')
   # Count the affected WT bond once even for double substitutions.
   f['WT_polar_contact_loss']-=edge['strength']*(1-retain_d*retain_a)
  for (p,r),dist in contact_pairs.items():
   if not (p in mutpos or r in mutpos):continue
   wp=ref[p-1];wr=ref[r-1];ap=sequence[p-1];ar=sequence[r-1]
   # Charge complementarity at observed WT contact is a weak compatibility prior.
   change=CHARGE[ap]*CHARGE[ar]-CHARGE[wp]*CHARGE[wr]
   f['contact_charge']-=.95*(bur[p]+bur[r])/2*math.exp(-max(0,dist-3.5))*change
   if p in mutpos and r in mutpos:
    vp=VOL[ap]-VOL[wp];vr=VOL[ar]-VOL[wr]
    if vp>0 and vr>0:
     f['joint_packing']-=.00006*bur[p]*bur[r]*vp*vr
  for (p,r),packet in pairs.items():
   if p in mutpos and r in mutpos:
    en=packet['enrichment'];wt=en[GROUP[ref[p-1]],GROUP[ref[r-1]]];alt=en[GROUP[sequence[p-1]],GROUP[sequence[r-1]]]
    f['weak_natural_covariation']+=.10*np.clip(alt-wt,-2,2)
  # No measured WT or neutral anchor: zero is simply the physical reference prior.
  total=float(sum(f.values()))
  score=round(math.tanh(total/4),6)
  assert math.isfinite(score) and -1<=score<=1
  preds.append({'id':q['id'],'score':score})
  score_packet=mi[q['id']]['scores']['sum_log_frequency_ratio']
  diagnostics.append({'id':q['id'],'mutation':q['mutation'],'features':dict(f),'raw_score':total,'msa_observed_terms':score_packet['observed_terms'],'msa_required_terms':score_packet['required_terms'],'complete_supplied_MSA_score':score_packet['score'],'WT_structure_coverage':1.0})
 summary='Unsupervised stability prior combines WT burial/packing, backbone propensity and polar-contact chemistry with weak homolog marginals and shrunk contact covariation. Source hashes and exact mutation/residue mapping verified; coordinates cover 47/47 residues, while 8 sites lack MSA support (missing terms omitted; partial sums not treated as complete). No fixed partners supplied. One global tanh scale, higher is better. Uncalibrated: no assay outcomes, mutant relaxation or validated epistasis; close rankings are uncertain.'
 out={'predictions':preds,'summary':summary}
 assert len(preds)==250 and len({p['id'] for p in preds})==250
 json.dump(out,open(ROOT/'final_predictions.json','w'),indent=2)
 json.dump({'model':'fixed unsupervised stability prior','hbonds':edges,'natural_contact_pairs':{f'{p}-{r}':{'effective_depth':d['coverage_depth']} for (p,r),d in pairs.items()},'candidates':diagnostics},open(ROOT/'model_diagnostics.json','w'),indent=2)
 print('Scores saved:',len(preds),'range',min(p['score'] for p in preds),max(p['score'] for p in preds))
 for label,ranked in [('top',sorted(diagnostics,key=lambda d:-d['raw_score'])[:12]),('bottom',sorted(diagnostics,key=lambda d:d['raw_score'])[:12])]:
  print(label,[(d['id'],d['mutation'],round(d['raw_score'],2)) for d in ranked])
 print('WT polar edges',[(d['donor_pos'],d['donor_atom'],d['acceptor_pos'],d['acceptor_atom'],round(d['strength'],2)) for d in edges])
if __name__=='__main__':predict()
