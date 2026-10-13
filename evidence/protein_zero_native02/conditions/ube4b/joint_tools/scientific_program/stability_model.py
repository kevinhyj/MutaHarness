"""Transparent zero-shot stability prior for the supplied UBE4B candidates.
Inputs are only public sequences, alignment, and static WT coordinates.
Coefficients are fixed chemistry priors, not fitted or selected on outcomes.
Geometry terms are compatibility heuristics, not calculated mutant energies.
"""
from pathlib import Path
from collections import defaultdict
import json,math,re,hashlib,itertools
import numpy as np

ROOT=Path(__file__).resolve().parent
load=lambda n:json.loads((ROOT/n).read_text())
case=load('case.json');manifest=load('evidence_manifest.json');msa=load('msa_evidence.json')
structure=load('structure_evidence.json');profile=load('local_homolog_profile.json')
ref=case['reference_sequence'];raw=structure['raw_coordinate_evidence']
sites={s['position']:s for s in case['evidence']['sites']}
rs={s['position']:s for s in raw['sites']};ps={int(p):s for p,s in profile['sites'].items()}
fixed_msa={x['id']:x for x in msa['candidates']}
assert manifest['support_count']==0 and not manifest['measurements_present']
assert case['context']['higher_is_better'] and case['context']['endpoint']=='Stability'
assert not case.get('reference_chains') # This case has one chain and no partners.
refhash=hashlib.sha256(ref.encode()).hexdigest()
assert refhash==profile['reference_sha256']==structure['reference_sequence_sha256']==msa['reference_sequences_sha256']['__single__']
source_checks={}
for path,spec in manifest['files'].items():
 if isinstance(spec,dict) and 'sha256' in spec:
  actual=hashlib.sha256((ROOT/path).read_bytes()).hexdigest();assert actual==spec['sha256'],path
  source_checks[path]=actual
assert raw['raw_source']['sha256']==source_checks['raw_evidence/UBE4B_HUMAN.pdb']
assert profile['msa_sha256']==source_checks['raw_evidence/UBE4B_HUMAN_2023-08-07_b04.a2m']
assert raw['mapping'][0]['method']=='exact_whole_chain_sequence' and raw['mapping'][0]['mapped_observed_residues']==len(ref)
assert all(s['backbone_complete'] and s['sidechain_complete'] and s['coordinate_present'] for s in rs.values())
AA='ACDEFGHIKLMNPQRSTVWY'
hyd=dict(zip(AA,[.31,1.54,-.77,-.64,1.79,0,.13,1.80,-.99,1.70,1.23,-.60,.72,-.22,-1.01,-.04,.26,1.22,2.25,.96]))
volume=dict(zip(AA,[88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.0,116.1,140.0,227.8,193.6]))
helix_energy=dict(zip(AA,[0,.68,.69,.40,.54,1.00,.61,.41,.26,.21,.24,.65,3.0,.39,.21,.50,.66,.61,.49,.53]))
positive_phi=dict(zip(AA,[1.0,.5,.3,1.0,1.6,0,.8,2.0,1.0,1.5,1.3,.2,3.0,1.0,1.0,.4,1.3,2.0,1.8,1.6]))
charge=lambda a:-1 if a in 'DE' else 1 if a in 'KR' else 0
burial={p:1-min(1,max(0,s['isolated_chain_wt_RSA'])) for p,s in rs.items()}
# Origin is caller-asserted AF2, not independently established; confidence only
# tempers geometry terms. It is not included as a variant fitness predictor.
confidence={p:min(1,max(0,(s['source_asserted_AF2_pLDDT']-50)/40)) for p,s in rs.items()}
atoms=defaultdict(dict)
for line in (ROOT/'raw_evidence/UBE4B_HUMAN.pdb').read_text().splitlines():
 if line.startswith('ATOM') and line[76:78].strip()!='H' and float(line[54:60])>0:
  p=int(line[22:26]);assert line[21]=='A' and not line[16].strip()
  atoms[p][line[12:16].strip()]=np.array([float(line[30:38]),float(line[38:46]),float(line[46:54])])
assert len(atoms)==len(ref)
for p in atoms:assert np.allclose(atoms[p]['CA'],rs[p]['CA_coordinate_A'])
DONORS={'R':['NE','NH1','NH2'],'K':['NZ'],'N':['ND2'],'Q':['NE2'],'H':['ND1','NE2'],'S':['OG'],'T':['OG1'],'Y':['OH'],'W':['NE1'],'C':['SG']}
ACCEPTORS={'D':['OD1','OD2'],'E':['OE1','OE2'],'N':['OD1'],'Q':['OE1'],'H':['ND1','NE2'],'S':['OG'],'T':['OG1'],'Y':['OH'],'C':['SG']}
BACKBONE={'N','CA','C','O','OXT'}
# Collapse polar atom contacts to one contact per donor/acceptor residue pair.
# Backbone-backbone bonds are omitted except where a candidate introduces Pro;
# those effects are handled by backbone/secondary-structure priors below.
hbonds=[];salt=[]
for p,q in itertools.combinations(atoms,2):
 wp,wq=ref[p-1],ref[q-1]
 if charge(wp)*charge(wq)==-1:
  contacts=[]
  for a in DONORS.get(wp,[])+ACCEPTORS.get(wp,[]):
   for b in DONORS.get(wq,[])+ACCEPTORS.get(wq,[]):
    d=float(np.linalg.norm(atoms[p][a]-atoms[q][b]))
    if d<3.6:contacts.append(d)
  if contacts:
   salt.append({'p':p,'q':q,'distance':min(contacts)});continue
 for donor,acceptor in [(p,q),(q,p)]:
  wd,wa=ref[donor-1],ref[acceptor-1]
  da=DONORS.get(wd,[])+(['N'] if wd!='P' else [])
  aa=ACCEPTORS.get(wa,[])+['O']
  kinds={}
  for a in da:
   for b in aa:
    ds=a not in BACKBONE;ass=b not in BACKBONE
    if not ds and not ass:continue
    if abs(donor-acceptor)<=1 and (not ds or not ass):continue
    dist=float(np.linalg.norm(atoms[donor][a]-atoms[acceptor][b]))
    cutoff=3.4 if wd!='C' and wa!='C' else 3.6
    if dist<cutoff:
     kind=(ds,ass);kinds[kind]=min(dist,kinds.get(kind,float('inf')))
  for (ds,ass),dist in kinds.items():
   hbonds.append({'donor':donor,'acceptor':acceptor,'donor_sidechain':ds,'acceptor_sidechain':ass,'distance':dist})

# Two hydroxyls (or histidine/hydroxyl) can both be donor and acceptor, but the
# same close atom pair is only one putative contact. Retain alternative roles
# without counting reciprocal descriptions as two independent bonds.
unique_hbonds=[];symmetric={}
for h in hbonds:
 if h['donor_sidechain'] and h['acceptor_sidechain']:
  key=tuple(sorted((h['donor'],h['acceptor'])))
  if key in symmetric:
   prior=symmetric[key]
   prior.setdefault('alternative_directions',[]).append(h)
   continue
  symmetric[key]=h
 unique_hbonds.append(h)
hbonds=unique_hbonds

# Same chemistry preserves the role most confidently. Different polar shapes
# receive partial retention only; no replacement rotamer is constructed.
def retention(w,a,role):
 if w==a:return 1.
 groups=['DE','NQ','ST','KR']
 if any(w in g and a in g for g in groups):return .8
 valid=set('DENQHSTYC') if role=='acceptor' else set('RKNQHSTYWC')
 if a not in valid:return 0.
 if a=='C':return .2
 return .5

def parse(q):
 changes={};seq=list(ref)
 for term in q['mutation'].split(':'):
  match=re.fullmatch(r'([A-Z])(\d+)([A-Z])',term);assert match,term
  w,p,a=match.group(1),int(match.group(2)),match.group(3)
  assert ref[p-1]==w and p not in changes and a in AA and a!=w
  changes[p]=a;seq[p-1]=a
 assert ''.join(seq)==q['sequence'] and len(changes)==q['mutation_order']
 return changes

def single_terms(p,a):
 w=ref[p-1];b=burial[p];c=confidence[p];s=sites[p]['wt_structure']
 phi,psi=s.get('phi_deg'),s.get('psi_deg')
 # Contiguous alpha-region torsions identify regular helices rather than turns.
 def alpha(pos):
  if pos not in sites:return False
  t=sites[pos]['wt_structure'];ph=t.get('phi_deg');pp=t.get('psi_deg')
  return ph is not None and pp is not None and -100<ph<-30 and -80<pp<0
 lo=hi=p
 if alpha(p):
  while alpha(lo-1):lo-=1
  while alpha(hi+1):hi+=1
 helix=alpha(p) and hi-lo+1>=4
 dh=hyd[a]-hyd[w];dv=(volume[a]-volume[w])/100
 terms={'mutation_burden':-.12,
        'hydrophobic_transfer':c*(1.15*b*dh-.20*(1-b)*max(0,dh)),
        'packing_mismatch':-c*b*(2.4*max(0,dv)**2+.9*max(0,-dv)**2),
        'backbone':0.,'evolutionary':0.}
 if w=='P':
  # Native prolines often cap helices or stabilize turns. Conventional
  # helix propensity is not an appropriate removal bonus for those sites.
  terms['backbone']-=.7*c
 elif helix:
  delta=helix_energy[a]-helix_energy[w]
  if a=='P' and p==lo:delta*=.65 # N1 is less disruptive than interior Pro.
  terms['backbone']-=c*delta
 else:
  if a=='G' and w!='G':terms['backbone']-=.5*c
  if w=='G' and a!='G':terms['backbone']+=.25*c
  if a=='P' and w!='P':
   # Proline strongly restricts phi and removes a backbone NH.
   phi_penalty=max(0,abs((phi if phi is not None else -65)+65)-25)/35
   terms['backbone']-=c*(.65+min(2.5,phi_penalty))
 if phi is not None and phi>0:
  terms['backbone']-=1.4*c*(positive_phi[a]-positive_phi[w])
 if p in ps and ps[p]['effective_non_gap_depth']>0:
  m=ps[p];f=m['frequencies'];reliability=m['effective_non_gap_depth']/(m['effective_non_gap_depth']+20)
  # Mild robust truncation prevents pseudocount-floor frequencies from becoming
  # a falsely precise stability oracle. Frequencies themselves are unchanged.
  lr=math.log(f[a]/f[w])
  terms['evolutionary']=.24*reliability*min(3,max(-7,lr))
 return terms

def network_terms(changes):
 alt=lambda p:changes.get(p,ref[p-1]);hbloss=0.;saltchange=0.;packing=0.
 for h in hbonds:
  p,q=h['donor'],h['acceptor']
  if p not in changes and q not in changes:continue
  compatible=[]
  for direction in [h]+h.get('alternative_directions',[]):
   dp,ap=direction['donor'],direction['acceptor']
   dr=retention(ref[dp-1],alt(dp),'donor') if direction['donor_sidechain'] else float(alt(dp)!='P')
   ar=retention(ref[ap-1],alt(ap),'acceptor') if direction['acceptor_sidechain'] else 1.
   compatible.append(dr*ar)
  strength=.50*(.5+.5*(burial[p]+burial[q]))*min(confidence[p],confidence[q])
  hbloss-=strength*(1-max(compatible))
 for h in salt:
  p,q=h['p'],h['q']
  if p not in changes and q not in changes:continue
  strength=.85*(.5+.5*(burial[p]+burial[q]))*min(confidence[p],confidence[q])
  wt=charge(ref[p-1])*charge(ref[q-1]);new=charge(alt(p))*charge(alt(q))
  # Attraction weakened if different sidechain lengths are assumed to preserve
  # WT geometry. Charge reversals at an observed salt contact are penalized.
  retained=1.
  if new==-1:
   # Role is decided from ALT sign, allowing D/E and K/R conservative changes.
   retained=1.
   for t in (p,q):
    if alt(t)!=ref[t-1]:retained*=.8 if charge(alt(t))==charge(ref[t-1]) else .35
  saltchange-=strength*(new*retained-wt)
 # Small contact-based packing prior for two newly hydrophobic sidechains.
 # This is explicitly speculative; volume competition is also penalized.
 for p,q in itertools.combinations(changes,2):
  wp,wq=ref[p-1],ref[q-1];ap,aq=alt(p),alt(q)
  scp=[v for a,v in atoms[p].items() if a not in BACKBONE]
  scq=[v for a,v in atoms[q].items() if a not in BACKBONE]
  if not scp or not scq:continue
  distance=min(float(np.linalg.norm(x-y)) for x in scp for y in scq)
  contact=max(0,min(1,(6-distance)/2))
  if contact==0:continue
  b=math.sqrt(burial[p]*burial[q])*min(confidence[p],confidence[q])
  if ap in 'ACFILMVWY' and aq in 'ACFILMVWY' and wp not in 'FILMVWY' and wq not in 'FILMVWY':
   packing+=.45*contact*b
  # Overlarge substitutions occupying adjacent buried sidechain regions.
  dp=max(0,volume[ap]-volume[wp])/100;dq=max(0,volume[aq]-volume[wq])/100
  packing-=.8*contact*b*dp*dq
 return {'polar_contact_loss':hbloss,'salt_contact_change':saltchange,'paired_packing_prior':packing}

predictions=[];diagnostics=[]
for q in case['query']:
 changes=parse(q);components=defaultdict(float);singles=[]
 mt=fixed_msa[q['id']]['substitution_terms']
 assert {(t['position'],t['ALT']) for t in mt}==set(changes.items())
 for p,a in changes.items():
  t=single_terms(p,a);singles.append({'position':p,'ALT':a,'components':t,'local_evolution_observed':p in ps and ps[p]['effective_non_gap_depth']>0})
  for k,v in t.items():components[k]+=v
 for k,v in network_terms(changes).items():components[k]+=v
 value=sum(components.values());score=round(float(np.tanh(value/5)),7)
 assert math.isfinite(score) and -1<=score<=1
 predictions.append({'id':q['id'],'score':score})
 local_observed=sum(x['local_evolution_observed'] for x in singles)
 diagnostics.append({'id':q['id'],'mutation':q['mutation'],'components':dict(components),'raw_stability_prior':value,'single_terms':singles,'local_evolution_coverage':local_observed/len(changes),'local_complete_evolutionary_score':sum(x['components']['evolutionary'] for x in singles) if local_observed==len(changes) else None,'supplied_msa_complete':fixed_msa[q['id']]['scores']['sum_log_frequency_ratio']['complete']})
ids=[q['id'] for q in case['query']]
assert len(ids)==len(set(ids))==250 and set(ids)=={f'M{i:03d}' for i in range(1,251)}
assert [p['id'] for p in predictions]==ids
summary=('Zero-shot stability prior combining closer-homolog conservation (2,890 unique rows; weighted Neff 577) with WT burial, sidechain size, backbone compatibility and putative polar-contact preservation. Closer homologs reduce the supplied broad profile\'s RING-like cysteine bias. All source hashes and 69-residue mapping verified; absent evolutionary terms remain missing. No fixed partners; one global tanh scale, higher is better. Uncalibrated: WT geometry, functional conservation and speculative paired packing cannot establish mutant stability or epistasis.')
output={'predictions':predictions,'summary':summary}
(ROOT/'final_predictions.json').write_text(json.dumps(output,indent=2)+'\n')
(ROOT/'prediction_diagnostics.json').write_text(json.dumps({'source_checks':source_checks,'reference_hash':refhash,'local_profile_selection':profile['selection'],'local_profile_neff':profile['neff'],'raw_coordinate_coverage':len(rs),'putative_hbonds':hbonds,'putative_salt_contacts':salt,'candidates':diagnostics},indent=2)+'\n')
print(json.dumps({'n':len(predictions),'range':[min(x['score'] for x in predictions),max(x['score'] for x in predictions)],'local_complete':sum(x['local_evolution_coverage']==1 for x in diagnostics),'salt_contacts':salt}))
for d in sorted(diagnostics,key=lambda x:x['raw_stability_prior'])[:8]+sorted(diagnostics,key=lambda x:-x['raw_stability_prior'])[:12]:
 print(d['id'],d['mutation'],round(d['raw_stability_prior'],3),{k:round(v,2) for k,v in d['components'].items()})
