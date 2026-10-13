"""Zero-shot GFP fluorescence ranking; no assay labels or fitted coefficients.
Run: python predict.py --stage checkpoint|final
"""
import argparse, hashlib, json, math, re
from pathlib import Path
import numpy as np

AA='ARNDCQEGHILKMFPSTWYV'
BLOSUM=np.array([
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
INDEX={a:i for i,a in enumerate(AA)}
MAXASA={'A':129,'R':274,'N':195,'D':193,'C':167,'Q':225,'E':223,'G':104,'H':224,'I':197,'L':201,'K':236,'M':224,'F':240,'P':159,'S':155,'T':172,'W':285,'Y':263,'V':174}
VOLUME=dict(zip('ACDEFGHIKLMNPQRSTVWY',[88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.,116.1,140.,227.8,193.6]))
HYDRO=dict(zip('ACDEFGHIKLMNPQRSTVWY',[1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]))
CHARGE={a:1 if a in 'KR' else -1 if a in 'DE' else 0 for a in AA}

def load_inputs():
 c=json.load(open('case.json')); m=json.load(open('evidence_manifest.json')); s=json.load(open('structure_evidence.json'))
 assert m['support_count']==0 and not m['measurements_present']
 assert c['context']['higher_is_better']
 wt=c['reference_sequence']; whash=hashlib.sha256(wt.encode()).hexdigest()
 assert whash==s['reference_sequence_sha256']
 source_checks={}
 for path,item in m['files'].items():
  if isinstance(item,dict) and 'sha256' in item:
   digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
   assert digest==item['sha256'],path
   source_checks[path]=digest
 assert s['raw_coordinate_evidence']['reference_sequence_sha256']==whash
 assert s['raw_coordinate_evidence']['raw_source']['sha256']==source_checks['raw_evidence/GFP_AEQVI.pdb']
 rows={x['id']:x for x in s['candidates']}
 assert len(rows)==250
 expected_ids={f'M{i:03d}' for i in range(1,251)}
 assert {x['id'] for x in c['query']}==expected_ids and len(c['query'])==250
 for q in c['query']:
  seq=list(wt); seen=set()
  for token in q['mutation'].split(':'):
   match=re.fullmatch(r'([A-Z])(\d+)([A-Z])',token); assert match,token
   a,p,b=match.groups(); p=int(p)
   assert wt[p-1]==a and p not in seen
   seen.add(p);seq[p-1]=b
  assert ''.join(seq)==q['sequence'] and len(seen)==q['mutation_order']
  r=rows[q['id']]; assert r['mutation']==q['mutation'] and r['mutation_count']==len(seen)
  assert {x['position'] for x in r['sites']}==seen
  for x in r['sites']:
   assert x['chain'] is None and x['WT']==wt[x['position']-1] and x['ALT']==q['sequence'][x['position']-1]
 # This fluorescence case supplies no fixed partner chains; never synthesize partners.
 assert not c.get('reference_chains') and not c.get('partners')
 return c,m,s,rows,source_checks

def checkpoint_site(x):
 a,b,p=x['WT'],x['ALT'],x['position']
 if a==b:return 0.
 rsa=x['derived_descriptors'].get('wt_rsa'); burial=0.5 if rsa is None else float(np.exp(-max(0,rsa)/.18))
 z=float(BLOSUM[INDEX[a],INDEX[b]])
 cost=.22+.10*max(0,2-z)+burial*(.25+.012*abs(VOLUME[b]-VOLUME[a])+.12*max(0,HYDRO[a]-HYDRO[b]))
 phi=x['wt_observations'].get('phi_deg'); psi=x['wt_observations'].get('psi_deg')
 if b=='P':cost+=1.4 if phi is not None and (phi < -90 or phi> -30 or (psi is not None and -80<psi<0)) else .45
 if a=='G' and phi is not None and phi>0:cost+=1.8
 if a=='P':cost+=.35
 if p in (66,67):cost+=5.
 if p in (96,222):cost+=2.5
 return cost

def dihedral(a,b,c,d):
    b0=-(b-a); b1=c-b; b2=d-c; b1=b1/np.linalg.norm(b1)
    v=b0-np.dot(b0,b1)*b1; w=b2-np.dot(b2,b1)*b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1,v),w),np.dot(v,w))))


def structural_features(case, packet):
    # Use source-bound complete raw WT atoms, not the sparse nearest-three graph.
    # None is retained as missing; pLDDT only weights reliability of geometry.
    from structure_tools import _read_pdb
    residues=_read_pdb(Path('raw_evidence/GFP_AEQVI.pdb'))
    assert len(residues)==len(case['reference_sequence'])
    assert ''.join(r['WT'] for r in residues)==case['reference_sequence']
    assert all(r['pdb_chain']=='A' for r in residues)
    obs={x['position']:x for x in packet['raw_coordinate_evidence']['sites']}
    assert all(obs[i+1]['pdb_residue_number']==r['pdb_residue_number'] for i,r in enumerate(residues))
    coords={i+1:{k:np.array(a['coord']) for k,a in r['atoms'].items() if a['element']!='H'} for i,r in enumerate(residues)}
    chrom=np.array([xyz for p in (65,66,67) for xyz in coords[p].values()])
    features={}
    for p,atoms in coords.items():
        o=obs[p]
        assert o['WT']==case['reference_sequence'][p-1] and o['coordinate_present']
        atoms_arr=np.array(list(atoms.values()))
        chrom_distance=float(np.linalg.norm(atoms_arr[:,None]-chrom[None,:],axis=2).min())
        omega=None; nh_acceptor=False
        if p>1 and all(k in atoms for k in ('N','CA')) and all(k in coords[p-1] for k in ('C','CA')):
            n,ca,cprev=atoms['N'],atoms['CA'],coords[p-1]['C']
            omega=dihedral(coords[p-1]['CA'],cprev,n,ca)
            # Approximate virtual NH direction in the WT peptide plane, only a
            # compatibility descriptor for potential loss upon introducing P.
            u=(n-cprev)/np.linalg.norm(n-cprev)+(n-ca)/np.linalg.norm(n-ca)
            h=n+1.01*u/np.linalg.norm(u)
            for other,at in coords.items():
                if abs(other-p)<=2 or 'O' not in at:continue
                no=at['O']-n
                if np.linalg.norm(no)<3.5 and np.linalg.norm(at['O']-h)<2.6 and np.dot(no/np.linalg.norm(no),u/np.linalg.norm(u))>.7:
                    nh_acceptor=True;break
        features[p]={'rsa':o['isolated_chain_wt_RSA'], 'plddt':o['source_asserted_AF2_pLDDT'],
                     'sidechain_complete':o['sidechain_complete'], 'backbone_complete':o['backbone_complete'],
                     'omega_deg':omega,'WT_NH_acceptor_compatible':nh_acceptor,
                     'chromophore_WT_heavy_distance_A':chrom_distance}
    return features

# Small priors for established GFP engineering substitutions; these are
# background biological knowledge, never measurements supplied for this assay.
ENGINEERING_PRIOR={('S',30,'R'):-.40,('Y',39,'N'):-.55,('F',99,'S'):-.45,
                  ('N',105,'T'):-.35,('Y',145,'F'):-.35,('M',153,'T'):-.50,
                  ('V',163,'A'):-.40,('I',171,'V'):-.40,('A',206,'V'):-.10}


def final_site(x,features):
    a,b,p=x['WT'],x['ALT'],x['position']; f=features[p]
    if a==b:return 0.,{'identity':0.}
    rsa=f['rsa']
    # Exposure decays continuously. Missing RSA would receive a generic prior,
    # not zero burial; all sites here have observed complete-sidechain RSA.
    burial=.5 if rsa is None else float(np.exp(-max(0.,rsa)/.20))
    pl=f['plddt']; reliability=.75 if pl is None else float(np.clip((pl-50)/40,.35,1.))
    z=float(BLOSUM[INDEX[a],INDEX[b]])
    phi=x['wt_observations'].get('phi_deg'); psi=x['wt_observations'].get('psi_deg')
    alpha=phi is not None and psi is not None and -100<phi<-30 and -80<psi<10
    extended=phi is not None and psi is not None and phi<-85 and (psi>65 or psi<-120)
    regular=alpha or extended
    polar=lambda aa:aa in 'STNQDEKRHY'
    aromatic=lambda aa:aa in 'FWYH'
    terms={'mutation_prior':.18, 'substitution':(.07+.06*burial)*max(0.,2-z)}
    packing=.009*abs(VOLUME[b]-VOLUME[a])+.105*max(0.,HYDRO[a]-HYDRO[b])
    packing+=.30*(polar(a)!=polar(b))+.35*(aromatic(a) and not aromatic(b))
    terms['packing']=reliability*burial*packing
    ca,cb=CHARGE[a],CHARGE[b]
    charge=1.55 if ca==0 and cb!=0 else .75 if ca!=0 and cb==0 else 1.45 if ca*cb<0 else 0.
    terms['buried_charge']=reliability*burial*charge
    backbone=0.
    if b=='P':
        incompatible_phi=phi is not None and not (-90<phi<-35)
        backbone+=.40+1.10*regular+.85*incompatible_phi+.45*burial+.55*f['WT_NH_acceptor_compatible']
    if a=='P':
        backbone+=.45
        if f['omega_deg'] is not None and abs(f['omega_deg'])<45:backbone+=1.85
    if a=='G' and phi is not None and phi>20:
        backbone+=1.35 if b in 'NSD' else 1.65 if b=='A' else 2.15
    if b=='G':backbone+=.50*alpha+.25*extended
    terms['backbone']=reliability*backbone
    # WT pre-chromophore geometry only suggests local photophysical sensitivity;
    # it is not the mature covalently linked chromophore or a mutant structure.
    d=f['chromophore_WT_heavy_distance_A']
    proximity=.40 if d<3.5 else .22 if d<5.5 else 0.
    terms['chromophore_environment']=proximity*min(1.5,max(0.,2-z)/4.)
    functional=0.
    if p==96 and b!='R':functional=2.3 if b=='K' else 3.5 if b=='H' else 5.5
    if p==222 and b not in 'DE':functional=.65 if b in 'QG' else 1.05
    if p==205 and b not in 'ST':functional=.70
    if p==203 and b not in 'ST':functional=.25
    terms['maturation_photophysics']=functional
    cost=sum(terms.values())
    if p==65:
        if b=='P':cost=max(cost,9.)
        elif b=='T':cost=-.65
        elif b in 'ACG':cost=min(cost,.75)
        else:cost=max(cost,3.8)
    if p==66:cost=max(cost,6.5 if b in 'FWH' else 10.)
    if p==67:cost=max(cost,8.)
    known=ENGINEERING_PRIOR.get((a,p,b))
    if known is not None:
        # Modest expected benefit, with strong background dependence acknowledged.
        cost=known
    terms['functional_floor_or_engineering_adjustment']=cost-sum(terms.values())
    return cost,terms


def main():
 parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['checkpoint','final'],default='checkpoint');args=parser.parse_args()
 c,m,s,rows,hashes=load_inputs()
 predictions=[];details=[]
 features=structural_features(c,s) if args.stage=='final' else None
 for q in c['query']:
  if args.stage=='final':
   estimates=[final_site(x,features) for x in rows[q['id']]['sites']]
   costs=[e[0] for e in estimates]; terms=[e[1] for e in estimates]
  else:
   costs=[checkpoint_site(x) for x in rows[q['id']]['sites']];terms=None
  burden=sum(costs)
  # Global monotonic bounded transform, no within-block normalization, calibrated
  # WT fluorescence, or probability interpretation. Arctan avoids floor ties.
  score=float(2/np.pi*np.arctan((3.5-burden)/2.5)) if args.stage=='final' else float(np.tanh((3.-burden)/3.))
  predictions.append({'id':q['id'],'score':round(score,8)})
  details.append({'id':q['id'],'mutation':q['mutation'],'costs':costs,'terms':terms,'burden':burden,'score':score})
 summary=('Zero-shot GFP fluorescence ranking from additive mutation burden, BLOSUM62/physicochemical compatibility, WT burial and backbone geometry (including cis-Pro89), chromophore/maturation biology, and modest established GFP engineering priors. Source hashes, exact 238-residue mapping and all candidate mutations verified; all queried sites have complete WT atoms and exposure evidence. No MSA, assay outcomes or fixed partners. One global monotonic scale; scores are uncalibrated, with substantial uncertainty from static pre-chromophore geometry, spectral effects and unmodeled epistasis.' if args.stage=='final' else 'Unsupervised GFP fluorescence prior: mutation burden, BLOSUM62 physicochemical changes, WT solvent burial and backbone compatibility, plus chromophore/maturation-site penalties. Verified mutation identities, sequence mapping and source hashes. No measured outcomes or MSA; static WT geometry cannot predict mutant relaxation or epistasis. One global monotonic score scale; substantial uncertainty.')
 out={'predictions':predictions,'summary':summary}
 assert len(predictions)==250 and len({x['id'] for x in predictions})==250
 assert all(math.isfinite(x['score']) and -1<=x['score']<=1 for x in predictions)
 target='final' if args.stage=='final' else 'checkpoint'
 Path(target+'_predictions.json').write_text(json.dumps(out,indent=2)+'\n')
 Path(target+'_diagnostics.json').write_text(json.dumps({'source_hashes':hashes,'structural_features':features,'details':details},indent=2)+'\n')
 print(target.capitalize(),'saved:',len(predictions),'scores; range',min(x['score'] for x in predictions),max(x['score'] for x in predictions))
if __name__=='__main__':main()
