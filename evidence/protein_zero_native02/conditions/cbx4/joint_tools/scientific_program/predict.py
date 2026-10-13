"""Label-free stability prior; no outcome fitting or external data."""
import json, math, re, hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import structure_tools

ROOT = Path(__file__).resolve().parent
HYDRO = dict(zip('ARNDCQEGHILKMFPSTWYV',
    [.31,-1.01,-.60,-.77,1.54,-.22,-.64,0,.13,1.8,1.7,-.99,1.23,1.79,.72,-.04,.26,2.25,.96,1.22]))
VOLUME = dict(zip('ARNDCQEGHILKMFPSTWYV',
    [88.6,173.4,114.1,111.1,108.5,143.8,138.4,60.1,153.2,166.7,166.7,168.6,162.9,189.9,112.7,89,116.1,227.8,193.6,140]))
# Experimental helix-propensity scale (Pace/Scholtz), used as a prior,
# not as CBX4 target-assay supervision or an absolute free-energy prediction.
HELIX_COST = dict(zip('ARNDCQEGHILKMFPSTWYV',
    [0,.21,.65,.69,.68,.39,.40,1.,.61,.41,.21,.26,.24,.54,3.16,.50,.66,.49,.53,.61]))
DONORS = {'R':['NE','NH1','NH2'],'K':['NZ'],'N':['ND2'],'Q':['NE2'],
          'S':['OG'],'T':['OG1'],'Y':['OH'],'W':['NE1'],'H':['ND1','NE2']}
ACCEPTORS = {'D':['OD1','OD2'],'E':['OE1','OE2'],'N':['OD1'],'Q':['OE1'],
             'S':['OG'],'T':['OG1'],'Y':['OH'],'H':['ND1','NE2']}
CHARGE = {'R':1,'K':1,'D':-1,'E':-1}

def load(name):
    return json.loads((ROOT/name).read_text())

def identity_and_sources(case, manifest, msa, structure):
    queries = case['query']; wt = case['reference_sequence']
    assert case['context']['higher_is_better'] is True
    assert len(queries)==250 and len({q['id'] for q in queries})==250
    assert set(q['id'] for q in queries)=={f'M{i:03d}' for i in range(1,251)}
    digest=hashlib.sha256(wt.encode()).hexdigest()
    assert msa['reference_sequences_sha256']['__single__']==digest
    assert structure['reference_sequence_sha256']==digest
    hashes={}
    for name, meta in manifest['files'].items():
        if isinstance(meta,dict) and 'sha256' in meta:
            p=ROOT/name
            hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
            if p.exists(): assert hashes[name]==meta['sha256'], name
    for q in queries:
        seq=list(wt); seen=set()
        for term in q['mutation'].split(':'):
            match=re.fullmatch(r'([A-Z])(\d+)([A-Z])',term)
            assert match
            a,p,b=match.groups(); p=int(p)
            assert wt[p-1]==a and p not in seen
            seen.add(p); seq[p-1]=b
        assert ''.join(seq)==q['sequence']
        assert q['mutation_order']==len(seen)
    assert not case.get('partners'), 'Partner preservation needs explicit handling'
    return hashes

def initial_effect(site, term):
    a,b=site['WT'],site['ALT']
    if a==b: return 0.,{}
    rsa=site['derived_descriptors']['wt_rsa']
    assert rsa is not None
    burial=max(0.,min(1.,1-rsa/.65))
    obs=site['wt_observations']; phi,psi=obs['phi_deg'],obs['psi_deg']
    helix=phi is not None and psi is not None and -100<phi<-30 and -85<psi<5
    beta=phi is not None and psi is not None and phi<-65 and (psi>70 or psi<-140)
    cost=1.4*burial*(HYDRO[a]-HYDRO[b])
    dv=VOLUME[b]-VOLUME[a]
    cost+=burial*(.008*max(-dv,0)+.006*max(dv,0))
    if b=='P' and (helix or beta): cost+=2.8
    if b=='G' and (helix or beta): cost+=1.0
    if a=='G' and phi is not None and phi>0: cost+=2.1
    if a=='P': cost+=.6
    lr=term['sum_log_frequency_ratio']
    evo=0.
    if lr is not None and term['non_gap_depth']>0:
        evo=.18*(.3+.7*burial)*3*math.tanh(lr/3)
    return -cost+evo, {'burial':burial,'physical_cost':cost,'evolutionary_effect':evo,'msa_present':lr is not None}

def geometry(case, structure):
    residues=structure_tools._read_pdb(ROOT/'raw_evidence/CBX4_HUMAN.pdb')
    assert len(residues)==50 and ''.join(r['WT'] for r in residues)==case['reference_sequence']
    assert {r['pdb_chain'] for r in residues}=={'A'}
    assert all(r['pdb_residue_number']==i for i,r in enumerate(residues,1))
    raw=structure['raw_coordinate_evidence']
    assert raw['raw_source']['matches_declared_source_sha256']
    assert raw['graph_scope']['complete_for_selected_observed_atoms_at_cutoff']
    atom_list=[a for r in residues for a in r['atoms'].values()]
    areas=structure_tools._sasa_arrays(np,atom_list,100,1.4)
    atom_asa={id(a):v for a,v in zip(atom_list,areas)}
    polar_exposure={}; backbone_bonds={i:[] for i in range(1,51)}
    bonds={}
    def dist(x,y): return float(np.linalg.norm(np.array(x['coord'])-np.array(y['coord'])))
    for i,r in enumerate(residues,1):
        atoms=list(r['atoms'].values())
        local_areas=structure_tools._sasa_arrays(np,atoms,100,1.4)
        local={a['name']:v for a,v in zip(atoms,local_areas)}
        group=set(DONORS.get(r['WT'],[])+ACCEPTORS.get(r['WT'],[]))
        denominator=sum(local[x] for x in group)
        polar_exposure[i]=min(1.,sum(atom_asa[id(r['atoms'][x])] for x in group)/denominator) if denominator>0 else None
        # Reproduce supplied SASA; no coordinates for any mutant are generated.
        total=sum(atom_asa[id(a)] for a in atoms)
        assert abs(total-raw['sites'][i-1]['isolated_chain_observed_atom_SASA_A2'])<1e-6
        for dn in DONORS.get(r['WT'],[])+(['N'] if r['WT']!='P' else []):
            donor=r['atoms'][dn]
            hs=[x for x in atoms if x['element']=='H' and dist(donor,x)<1.25]
            for j,s in enumerate(residues,1):
                if i==j: continue
                for an in ACCEPTORS.get(s['WT'],[])+['O']:
                    acc=s['atoms'][an]; d=dist(donor,acc)
                    if d>3.6: continue
                    directional=False
                    for h in hs:
                        v=np.array(donor['coord'])-h['coord']; w=np.array(acc['coord'])-h['coord']
                        cosine=float(np.dot(v,w)/np.linalg.norm(v)/np.linalg.norm(w))
                        if dist(h,acc)<2.7 and cosine<-.5: directional=True
                    if not directional: continue
                    if dn=='N' and an=='O':
                        if abs(i-j)>=2: backbone_bonds[i].append(j)
                        continue
                    ionic=(CHARGE.get(r['WT'],0)*CHARGE.get(s['WT'],0)==-1 and dn!='N' and an!='O')
                    key=(i,j,dn=='N',an=='O')
                    bond={'donor':i,'acceptor':j,'donor_backbone':dn=='N','acceptor_backbone':an=='O',
                          'ionic':ionic,'distance':d,'value':1.15 if ionic else .65}
                    if key not in bonds or bond['distance']<bonds[key]['distance']: bonds[key]=bond
    # Require a sustained alpha-like run, rather than labeling isolated turn torsions as a helix.
    alpha=[]
    for s in case['evidence']['sites']:
        p=s['wt_structure']['phi_deg']; q=s['wt_structure']['psi_deg']
        alpha.append(p is not None and q is not None and -95<p<-35 and -75<q<5)
    helix=set(); run=[]
    for i,flag in enumerate(alpha+[False],1):
        if flag: run.append(i)
        else:
            if len(run)>=4 and any(j-4 in backbone_bonds[j] for j in run): helix.update(run)
            run=[]
    return {'helix':helix,'backbone_bonds':backbone_bonds,'polar_exposure':polar_exposure,'polar_bonds':list(bonds.values())}

def refined_effect(site,term,geom):
    a,b,p=site['WT'],site['ALT'],site['position']
    if a==b: return 0.,{}
    rsa=site['derived_descriptors']['wt_rsa']; assert rsa is not None
    burial=max(0.,min(1.,1-rsa/.65))
    phi=site['wt_observations']['phi_deg']; psi=site['wt_observations']['psi_deg']
    hydro=1.4*burial*(HYDRO[a]-HYDRO[b])
    # RSA of a whole charged residue also describes its buried aliphatic stem.
    # An exposed charged head is not an energetically buried ion to be rescued.
    if hydro<0 and a in CHARGE:
        head_exposure=geom['polar_exposure'][p]
        assert head_exposure is not None
        hydro*=.55*(1-head_exposure)
    elif hydro<0 and a in 'NQST':
        hydro*=.65
    dv=VOLUME[b]-VOLUME[a]
    packing=burial*(.014*max(-dv,0)+.009*max(dv,0))
    backbone=0.
    if p in geom['helix']:
        backbone+=HELIX_COST[b]-HELIX_COST[a]
    elif b=='G':
        backbone+=.55 if phi is not None and phi<0 else -.4
    if a=='G':
        if phi is not None and phi>0:
            backbone+=1.3 if b in 'DNST' else (3.5 if b=='P' else 2.4)
        elif p not in geom['helix']: backbone-=.35
    if a!='G' and phi is not None and phi>0:
        if b=='P': backbone+=3.5
        elif b=='G': backbone-=.3
        elif b in 'DNST': backbone-=.2
        elif b in 'IVTFWY': backbone+=.4
    if b=='P' and p not in geom['helix']:
        if geom['backbone_bonds'][p]: backbone+=1.5
        if phi is not None: backbone+=min(2.2,max(0.,(abs(phi+65)-25)/40))
        if not geom['backbone_bonds'][p] and phi is not None and -85<phi<-45: backbone-=.3
    if a=='P' and p not in geom['helix']: backbone+=.65
    evo=0.; lr=term['sum_log_frequency_ratio']
    if lr is not None and term['non_gap_depth']>0:
        evo=.22*(.4+.6*burial)*4*math.tanh(lr/4)
    effect=-hydro-packing-backbone+evo
    return effect,{'burial':burial,'hydrophobic_cost':hydro,'packing_cost':packing,
                  'backbone_cost':backbone,'evolutionary_effect':evo,'msa_present':lr is not None,
                  'complete_candidate_msa_not_assumed':True}

def interaction_loss(changes,case,geom):
    wt=case['reference_sequence']; loss=0.; details=[]
    for bond in geom['polar_bonds']:
        fractions=[]
        for role in ['donor','acceptor']:
            p=bond[role]; a=wt[p-1]; b=changes.get(p,a)
            if a==b: retained=1.
            elif bond[role+'_backbone']:
                retained=0. if role=='donor' and b=='P' else 1.
            elif bond['ionic']:
                retained=.8 if CHARGE.get(a)==CHARGE.get(b) else 0.
            else:
                table=DONORS if role=='donor' else ACCEPTORS
                retained=.5 if b in table else 0.
                if {a,b}<={'S','T'} or {a,b}<={'N','Q'} or {a,b}<={'D','E'}: retained=.8
            fractions.append(retained)
        cost=bond['value']*(1-fractions[0]*fractions[1])
        loss+=cost
        if cost: details.append({'donor':bond['donor'],'acceptor':bond['acceptor'],'cost':cost,'ionic':bond['ionic']})
    # Each observed WT interaction is counted once, even if both sites change.
    # Retention factors are uncertain chemistry priors, not mutant H-bond predictions.
    return loss,details

def main():
    case=load('case.json'); manifest=load('evidence_manifest.json')
    msa=load('msa_evidence.json'); structure=load('structure_evidence.json')
    hashes=identity_and_sources(case,manifest,msa,structure)
    geom=geometry(case,structure)
    ms={c['id']:c for c in msa['candidates']}
    ss={c['id']:c for c in structure['candidates']}
    preds=[]; diagnostics=[]
    for q in case['query']:
        m=ms[q['id']]; s=ss[q['id']]
        assert q['mutation']==m['mutation']==s['mutation']
        score=0.; ds=[]
        for site,term in zip(s['sites'],m['substitution_terms']):
            assert (site['position'],site['WT'],site['ALT'])==(term['position'],term['WT'],term['ALT'])
            effect,d=refined_effect(site,term,geom); score+=effect; ds.append(d)
        changes={site['position']:site['ALT'] for site in s['sites']}
        loss,bond_details=interaction_loss(changes,case,geom); score-=loss
        preds.append({'id':q['id'],'score':round(math.tanh(score/4),8)})
        diagnostics.append({'id':q['id'],'mutation':q['mutation'],'latent':score,'terms':ds,'WT_interaction_loss':loss,'bond_details':bond_details})
    summary=('Label-free stability prior: WT burial/packing, hydrophobic changes, backbone and helix propensities, '
             'loss of observed WT polar interactions, and weak bounded homolog log-frequency ratios. '
             'All source hashes and 50-residue coordinate mapping verified; structure covers all sites. '
             'MSA has zero depth at 11 positions: missing terms use physical priors, never complete partial MSA scores '
             '(147/250 candidates have complete MSA). No partners are supplied. One global tanh scale, higher is better. '
             'Uncalibrated ranking; static WT geometry and conservation are imperfect, with no mutant relaxation or validated epistasis.')
    result={'predictions':preds,'summary':summary}
    assert len(preds)==250 and all(math.isfinite(p['score']) and -1<=p['score']<=1 for p in preds)
    (ROOT/'checkpoint_predictions.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'final_predictions.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'prediction_diagnostics.json').write_text(json.dumps({'source_hashes':hashes,'helix_positions':sorted(geom['helix']),
        'WT_polar_contacts':geom['polar_bonds'],'candidates':diagnostics},indent=2)+'\n')
    print('Saved checkpoint_predictions.json; 250 finite unique scores; identities and source hashes verified.')
    print('Mutation counts:',dict(Counter(q['mutation_order'] for q in case['query'])))
    print('MSA complete:',sum(c['scores']['sum_log_frequency_ratio']['complete'] for c in msa['candidates']))
    print('Score range:',min(p['score'] for p in preds),max(p['score'] for p in preds))

if __name__=='__main__': main()
