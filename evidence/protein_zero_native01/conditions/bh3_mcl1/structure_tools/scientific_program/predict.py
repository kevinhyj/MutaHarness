"""Unsupervised BH3/Mcl-1 prioritization; no assay labels or fitted parameters.
Initial checkpoint: conservative physicochemical effects at observed WT interface.
"""
import json, re, math, hashlib
from pathlib import Path
import numpy as np

HYD=dict(zip('ACDEFGHIKLMNPQRSTVWY',[.31,1.54,-.77,-.64,1.79,0,.13,1.8,-.99,1.7,1.23,-.6,.72,-.22,-1.01,-.04,.26,1.22,2.25,.96]))
VOL=dict(zip('ACDEFGHIKLMNPQRSTVWY',[88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89,116.1,140,227.8,193.6]))
HELIX=dict(zip('ACDEFGHIKLMNPQRSTVWY',[0,.68,.69,.40,.54,1,.61,.41,.26,.21,.24,.65,3.16,.39,.21,.50,.66,.61,.49,.53]))

def load_inputs():
 c=json.load(open('case.json')); m=json.load(open('evidence_manifest.json')); e=json.load(open('structure_evidence.json')); r=e['raw_coordinate_evidence']
 assert not m['measurements_present'] and m['support_count']==0
 assert hashlib.sha256(c['reference_sequence'].encode()).hexdigest()==e['reference_sequence_sha256']==r['reference_sequence_sha256']
 path=Path('raw_evidence/3KZ0_hm.pdb')
 assert hashlib.sha256(path.read_bytes()).hexdigest()==m['files'][str(path)]['sha256']==r['raw_source']['sha256']
 sites={(s['chain'],s['position']):s for s in r['sites']}
 for (ch,p),s in sites.items(): assert s['WT']==c['reference_chains'][ch][p-1]
 ids=[q['id'] for q in c['query']]; assert len(ids)==len(set(ids))==250 and set(ids)=={f'M{i:03d}' for i in range(1,251)}
 for q in c['query']:
  chains={k:list(v) for k,v in c['reference_chains'].items()}; changes=[]
  for token in q['mutation'].split(';'):
   ch,wt,p,alt=re.fullmatch(r'(\w+):([A-Z])(\d+)([A-Z])',token).groups(); p=int(p)
   assert ch=='C' and chains[ch][p-1]==wt and wt!='X'
   chains[ch][p-1]=alt; changes.append((ch,p,wt,alt))
  assert len({(ch,p) for ch,p,_,_ in changes})==len(changes)==q['mutation_order']
  assert ''.join(chains['C'])==q['sequence']
  assert {k:''.join(v) for k,v in chains.items()}==q['mutated_chains']
  for ch,seq in c['context']['fixed_partner_chains'].items(): assert q['mutated_chains'][ch]==seq
  q['_changes']=changes
 return c,m,e,r,sites

def initial_effect(p,wt,a,s):
 if wt==a: return 0.
 burial=s['observed_atom_SASA_buried_by_context_A2']
 assert burial is not None
 w=min(1,burial/90)
 delta=0.45*(HELIX[wt]-HELIX[a])
 if p in (6,10,13,17):
  delta+=w*(1.05*(HYD[a]-HYD[wt])-.010*max(0,VOL[wt]-VOL[a])-.009*max(0,VOL[a]-VOL[wt]))
  if a in 'DEKR': delta-=1.2*w
 elif p==15:
  delta+= {'E':-1.,'N':-1.8,'Q':-2.2}.get(a,-3.8)
 elif p in (7,14):
  delta-=w*.024*max(0,VOL[a]-VOL[wt])
  if a in 'DEKR': delta-=1.0*w
 else:
  delta-=.20*w*abs(HYD[a]-HYD[wt])
  if wt in 'DEKR' and a not in 'DEKR': delta-=.15*w
 return delta

def output(c, values, path, summary):
 # One fixed monotonic transform for every candidate; no block normalization.
 result={'predictions':[{'id':q['id'],'score':round(float(math.tanh((v+3.)/5.)),6)} for q,v in zip(c['query'],values)],'summary':summary}
 assert len(result['predictions'])==250 and all(math.isfinite(x['score']) and -1<=x['score']<=1 for x in result['predictions'])
 Path(path).write_text(json.dumps(result,indent=2)+'\n')
 return result

def run_checkpoint():
 c,m,e,r,sites=load_inputs()
 values=[sum(initial_effect(p,w,a,sites[ch,p]) for ch,p,w,a in q['_changes']) for q in c['query']]
 summary='Unsupervised BH3/Mcl-1 interface heuristic: preserve hydrophobic anchors, small buried residues, the conserved Asp contact and helix propensity, with chain A fixed. Coordinate and reference hashes and mutation identities verified; peptide coordinate coverage 23/23; Mcl-1 unknown slots remain missing. No MSA or assay measurements. Static WT geometry cannot establish mutant affinity or epistasis. One global monotonic score scale; higher is predicted better.'
 out=output(c,values,'checkpoint_predictions.json',summary)
 print('Saved validated checkpoint:',len(out['predictions']),'range',min(p['score'] for p in out['predictions']),max(p['score'] for p in out['predictions']))


def geometry_features(c, r):
 """Observed WT pocket features. No mutant structures or optimized energies."""
 import structure_tools as st
 residues,mapping=st._map_raw_residues(st._prepare(c),st._read_pdb(Path('raw_evidence/3KZ0_hm.pdb'),include_zero_occupancy=True),None)
 assert mapping==r['mapping']
 lookup={(x['chain'],x['position']):x for x in residues if x['protein']}
 features={}
 # Standard donor/acceptor atom types. Histidine protonation is unknown:
 # contribute at half weight, and never assign it a definite positive charge.
 donor_names={'R':{'NE','NH1','NH2'},'K':{'NZ'},'N':{'ND2'},'Q':{'NE2'},'S':{'OG'},'T':{'OG1'},'Y':{'OH'},'W':{'NE1'},'H':{'ND1','NE2'}}
 acceptor_names={'D':{'OD1','OD2'},'E':{'OE1','OE2'},'N':{'OD1'},'Q':{'OE1'},'S':{'OG'},'T':{'OG1'},'Y':{'OH'},'H':{'ND1','NE2'}}
 for p in range(1,len(c['reference_sequence'])+1):
  residue=lookup['C',p]; atoms=residue['atoms']
  center=np.array(atoms.get('CB',atoms['CA'])['coord'])
  donors=[]; acceptors=[]
  for partner in residues:
   if partner.get('chain')!='A': continue
   aa=partner['WT']
   for name,atom in partner['atoms'].items():
    d=float(np.linalg.norm(center-np.array(atom['coord'])))
    # A reachability descriptor, not an actual mutant H-bond.
    # Short Asn/Asp/Ser/Thr sidechains can reach partners 4-5 A from CB.
    strength=max(0.,1.-abs(d-4.5)/1.5)
    if strength<=0: continue
    if name=='N' or name in donor_names.get(aa,set()): donors.append(strength*(.5 if aa=='H' and name!='N' else 1.))
    if name in ('O','OXT') or name in acceptor_names.get(aa,set()): acceptors.append(strength*(.5 if aa=='H' and name not in ('O','OXT') else 1.))
  features[p]={'donor_reach':max(donors,default=0.),'acceptor_reach':max(acceptors,default=0.)}
 # The ideal L-amino-acid beta-carbon direction at WT glycine indicates a
 # tightly restricted pocket. It is a template steric prior, not relaxation.
 residue=lookup['C',14]; n,ca,cc=[np.array(residue['atoms'][a]['coord']) for a in ('N','CA','C')]
 b=ca-n; v=cc-ca; virtual_cb=-.58273431*np.cross(b,v)+.56802827*b-.54067466*v+ca
 distances=[float(np.linalg.norm(virtual_cb-np.array(atom['coord']))) for partner in residues if partner.get('chain')=='A' for atom in partner['atoms'].values() if atom['element']!='H']
 features[14]['ideal_CB_nearest_partner_heavy_A']=min(distances)
 return features


def refined_effect(p,wt,a,s,f):
 if wt==a: return 0.
 isolated=s['isolated_chain_observed_atom_SASA_A2']; buried=s['observed_atom_SASA_buried_by_context_A2']
 assert isolated is not None and buried is not None and s['sidechain_complete']
 fraction=buried/max(isolated,1e-9)
 # Absolute interface area and fraction both matter; fully buried central
 # Leu10 receives greater contact weight than the more open Phe17 pocket.
 contact=.55*fraction+.45*min(1.2,buried/100.)
 helix=.65*(HELIX[wt]-HELIX[a])
 if a=='P': helix-=2.3 # backbone amide H-bond loss / kink in this WT helix
 effect=helix
 if p in (6,10,13,17):
  effect+=contact*(1.05*(HYD[a]-HYD[wt])-.012*max(0,VOL[wt]-VOL[a])-.009*max(0,VOL[a]-VOL[wt]))
  if a in 'DEKR': effect-=1.1*contact
  # Only partial credit for possible polar compensation: reachability cannot
  # establish orientation, protonation, desolvation or a relaxed H-bond.
  if a in 'NSTQHC':
   need_donor=a in 'NQH'; need_acceptor=a in 'NSTQHC'
   match=max(f['acceptor_reach'] if need_donor else 0.,f['donor_reach'] if need_acceptor else 0.)
   effect+=.75*contact*match
  elif a in 'DE': effect+=.65*contact*f['donor_reach']
  elif a in 'KR': effect+=.65*contact*f['acceptor_reach']
 elif p==15:
  # WT Asp makes a geometrically specific Arg92 salt bridge plus Asn89 H-bond.
  effect+= {'E':-.85,'N':-1.55,'Q':-1.95,'H':-2.9,'K':-4.0,'R':-4.0}.get(a,-3.3)
 elif p==14:
  # Every non-Gly adds a beta carbon into a constrained WT backbone pocket.
  clash=max(0.,3.2-f['ideal_CB_nearest_partner_heavy_A'])
  effect-=1.5*clash+.022*max(0,VOL[a]-VOL[wt])
  if a in 'DEKR': effect-=.7
 elif p==7:
  effect-=.026*max(0,VOL[a]-VOL[wt])
  if a in 'DEKR': effect-=.8
  if a in 'NSTQH': effect+=.25*max(f['donor_reach'],f['acceptor_reach'])
 elif p==11:
  # Native Arg donates to backbone O and Asp85, but its proximal atoms
  # lie close to Arg92. Short acidic residues could trade these contacts.
  # Shrunk toward neutrality because no relaxed mutant geometry is known.
  effect+= {'D':.35,'E':.15,'K':-.15,'N':-.25,'Q':-.2,'A':-.65,'G':-.6,'T':-.4,'L':-.55}.get(a,-.65)
 else:
  # Exposed E9/E16 chemistry is a weak prior; no core hydrophobic penalty.
  effect-=.12*contact*abs(HYD[a]-HYD[wt])
 return effect


def helix_charge_energy(seq):
 # Weak unsupervised helical i,i+3/i+4 electrostatic prior. This is not
 # experimentally established epistasis, and cannot override anchor losses.
 charge={'D':-1.,'E':-1.,'K':1.,'R':1.}
 return .10*sum(charge.get(seq[i],0)*charge.get(seq[j],0) for i in range(3,19) for j in (i+3,i+4) if j<19)


def run_refined():
 c,m,e,r,sites=load_inputs(); features=geometry_features(c,r)
 rows=[]; values=[]
 for q in c['query']:
  local=[{'position':p,'WT':w,'ALT':a,'effect':refined_effect(p,w,a,sites[ch,p],features[p])} for ch,p,w,a in q['_changes']]
  electro=helix_charge_energy(c['reference_sequence'])-helix_charge_energy(q['sequence'])
  value=sum(x['effect'] for x in local)+electro
  values.append(value); rows.append({'id':q['id'],'mutation':q['mutation'],'site_effects':local,'helix_charge_prior':electro,'raw_relative_score':value})
 summary='Unsupervised binding-specific BH3/Mcl-1 heuristic: WT interface burial, hydrophobic anchor packing, possible polar compensation, conserved Asp–Arg contact, glycine-pocket sterics and helix propensity; Mcl-1 chain A stays fixed. Source/reference hashes and mutation identities verified; peptide coordinates cover 23/23 residues, partner X25–31 remain unobserved. No MSA or assay labels. Static WT geometry and uncertain mutant relaxation/epistasis limit confidence; scores are relative priorities, not calibrated affinities. One global monotonic scale; higher is better.'
 result=output(c,values,'final_predictions.json',summary)
 Path('prediction_diagnostics.json').write_text(json.dumps({'geometry_features':features,'candidates':rows},indent=2)+'\n')
 print('Final validated',len(values),'range',min(p['score'] for p in result['predictions']),max(p['score'] for p in result['predictions']))
 for direction in [True,False]:
  print('best' if direction else 'worst')
  for q,v in sorted(zip(c['query'],values),key=lambda t:t[1],reverse=direction)[:8]: print(q['id'],q['mutation'],round(v,3))
 return result


if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--checkpoint',action='store_true',help='Reproduce the initial checkpoint heuristic')
 args=parser.parse_args()
 if args.checkpoint: run_checkpoint()
 else: run_refined()
