"""Unsupervised BH3/Mcl-1 ranking; saved checkpoint precedes structural refinement."""
import json, math, re, hashlib
from pathlib import Path


def load_inputs():
    case=json.load(open('case.json'))
    msa=json.load(open('msa_evidence.json'))
    structure=json.load(open('structure_evidence.json'))
    manifest=json.load(open('evidence_manifest.json'))
    # Verify bytes of public sources and reference identities before using evidence.
    for name,entry in manifest['files'].items():
        if isinstance(entry,dict) and 'sha256' in entry:
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==entry['sha256'], name
    for chain,seq in case['reference_chains'].items():
        assert hashlib.sha256(seq.encode()).hexdigest()==msa['reference_sequences_sha256'][chain]
    for q in case['query']:
        seqs=dict(case['reference_chains'])
        changes=[]
        for token in q['mutation'].split(';'):
            match=re.fullmatch(r'([A-Z]):([A-Z])(\d+)([A-Z])',token)
            assert match, token
            chain,wt,p,alt=match.groups();p=int(p)
            assert seqs[chain][p-1]==wt and wt!='X' and wt!=alt
            seqs[chain]=seqs[chain][:p-1]+alt+seqs[chain][p:]
            changes.append((chain,p))
        assert len(set(changes))==len(changes)==q['mutation_order']
        assert seqs==q['mutated_chains']
        assert seqs['C']==q['sequence']
        assert seqs['A']==case['context']['fixed_partner_chains']['A']
    assert len(case['query'])==250 and len({q['id'] for q in case['query']})==250
    assert case['context']['higher_is_better']
    return case,msa,structure


def checkpoint(case,msa):
    evidence={x['id']:x for x in msa['candidates']}
    predictions=[]
    for q in case['query']:
        row=evidence[q['id']];s=row['scores']['sum_log_frequency_ratio']
        assert s['complete'] and s['coverage']==1 and s['score'] is not None
        assert all(t['non_gap_depth']>0 for t in row['substitution_terms'])
        raw=s['score']
        # Proline is specifically disfavored in the bound BH3 helix.
        raw-=sum(3.0 for t in row['substitution_terms'] if t['ALT']=='P')
        predictions.append({'id':q['id'],'score':math.tanh(raw/14.0)})
    result={'predictions':predictions,'summary':'Initial unsupervised checkpoint: additive supplied natural-homolog log-frequency ratios, with a bound-helix proline penalty; one global tanh scale, higher is better. Source hashes, mutation identities, fixed Mcl-1 partner and complete nonzero-depth MSA coverage verified. No assay outcomes; conservation is an uncertain affinity prior and epistasis is unvalidated.'}
    Path('checkpoint_predictions.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

# Standard Fauchere-Pliska sidechain hydrophobicity, residue volumes (A^3),
# and Pace-Scholtz helix free-energy costs relative to alanine. These are
# physicochemical priors, not estimated assay coefficients.
HYDRO=dict(zip('ARNDCQEGHILKMFPSTWYV',
 [0.31,-1.01,-0.60,-0.77,1.54,-0.22,-0.64,0.0,0.13,1.80,1.70,-0.99,1.23,1.79,0.72,-0.04,0.26,2.25,0.96,1.22]))
VOLUME=dict(zip('ARNDCQEGHILKMFPSTWYV',
 [88.6,173.4,114.1,111.1,108.5,143.8,138.4,60.1,153.2,166.7,166.7,168.6,162.9,189.9,112.7,89.0,116.1,227.8,193.6,140.0]))
HELIX=dict(zip('ARNDCQEGHILKMFPSTWYV',
 [0,.21,.65,.69,.68,.39,.40,1.0,.61,.41,.21,.26,.24,.54,3.16,.50,.66,.49,.53,.61]))


def structure_features(case,structure):
    import numpy as np
    import structure_tools as st
    raw=structure['raw_coordinate_evidence']
    assert raw['raw_source']['matches_declared_source_sha256']
    residues,mapping=st._map_raw_residues(st._prepare(case),
        st._read_pdb(Path('raw_evidence/3KZ0_hm.pdb'),include_zero_occupancy=True),None)
    assert mapping==raw['mapping']
    coords={(r['chain'],r['position']):r for r in residues}
    sites={(r['chain'],r['position']):r for r in raw['sites']}
    # Deterministic outward-cone free-space descriptor of WT geometry.
    # This is neither a mutant rotamer nor a relaxed mutant structure.
    n=2048;i=np.arange(n);z=1-2*(i+.5)/n
    theta=np.pi*(3-np.sqrt(5))*i
    directions=np.column_stack((np.sqrt(1-z*z)*np.cos(theta),np.sqrt(1-z*z)*np.sin(theta),z))
    result={}
    for p in sorted({t['position'] for q in json.load(open('msa_evidence.json'))['candidates'] for t in q['substitution_terms']}):
        r=coords['C',p];obs=sites['C',p]
        assert obs['WT']==r['WT']==case['reference_sequence'][p-1]
        assert obs['backbone_complete'] and obs['sidechain_complete']
        assert obs['missing_standard_heavy_atoms']==[]
        ca=np.array(r['atoms']['CA']['coord'])
        side=[a['coord'] for a in r['atoms'].values() if a['element']!='H' and a['name'] not in ['N','CA','C','O','OXT']]
        if side:
            center=np.mean(side,axis=0)
        else:
            b=ca-np.array(r['atoms']['N']['coord']);c=np.array(r['atoms']['C']['coord'])-ca
            center=ca-.58273431*np.cross(b,c)+.56802827*b-.54067466*c
        axis=(center-ca)/np.linalg.norm(center-ca)
        points=ca+4.5*directions[directions@axis>.5]
        environment=[a for t in residues if t is not r for a in t['atoms'].values() if a['element']!='H']
        minimum=np.full(len(points),100.)
        for a in environment:minimum=np.minimum(minimum,np.linalg.norm(points-a['coord'],axis=1))
        openness=float(np.mean(minimum>2.8))
        partner_polar=[]
        for t in residues:
            if t['chain']!='A':continue
            for a in t['atoms'].values():
                if a['element'] in ['N','O'] and np.linalg.norm(center-a['coord'])<5.5:
                    partner_polar.append((t['position'],t['WT'],a['name']))
        iso=obs['isolated_chain_observed_atom_SASA_A2']
        burial=obs['observed_atom_SASA_buried_by_context_A2']
        assert iso is not None and burial is not None
        result[p]={'WT':r['WT'],'burial_fraction':min(1.,max(0.,burial/iso)),
            'buried_area_A2':burial,'outward_cone_open_fraction':openness,
            'nearby_partner_polar_atoms':partner_polar}
    return result,coords


def refined(case,msa,structure):
    import numpy as np
    features,coords=structure_features(case,structure)
    evidence={x['id']:x for x in msa['candidates']}
    contributions=[];predictions=[]
    for q in case['query']:
        row=evidence[q['id']];terms=[]
        for t in row['substitution_terms']:
            p,wt,alt=t['position'],t['WT'],t['ALT'];f=features[p]
            buried=f['burial_fraction'];openness=f['outward_cone_open_fraction']
            tightness=max(0.,1-openness/.30)
            # Bound peptide preorganization, without assuming generic folding fitness.
            helix=.75*(HELIX[alt]-HELIX[wt])
            if alt=='P' and ('C',p-4) in coords:
                donor=np.array(coords['C',p]['atoms']['N']['coord'])
                acceptor=np.array(coords['C',p-4]['atoms']['O']['coord'])
                if np.linalg.norm(donor-acceptor)<3.5:
                    # Proline removes this observed bound-helix NH donor;
                    # intrinsic helix propensity alone undercounts that loss.
                    helix+=1.5
            packing=0.;polar=0.
            if wt in 'AILVFM':
                packing=.80*buried*(HYDRO[wt]-HYDRO[alt])
                dv=VOLUME[alt]-VOLUME[wt]
                packing+=.004*buried*max(-dv,0)+.060*tightness*max(dv,0)
                if alt in 'FWY' and wt in 'AILV':packing+=.4*tightness
                if alt in 'AILVFM' and dv>0:
                    packing-=.006*dv*min(1.,max(0.,(openness-.2)/.3))
                if alt in 'DERK':polar+=1.5*buried
                if alt in 'NQSTHY':
                    # Only modest rescue for potential polar partners: WT proximity
                    # cannot establish a mutant hydrogen bond.
                    polar-=.3*min(2,len(f['nearby_partner_polar_atoms']))
            if wt=='G':
                # The profile and geometry both allow Ala at this small-residue site.
                packing+=.06*tightness*max(0,VOLUME[alt]-VOLUME['A'])
            if p==15:
                # Native Asp contacts fixed-partner Arg92 (2.77 A) and Asn89.
                polar+=0.55 if alt=='E' else (1.6 if alt in 'NQ' else 2.8)
                if alt in 'KR':polar+=1.5
            if p==11:
                # Native Arg hydrogen-bonds to backbone and Asp85. Its proximal
                # CG also lies 3.31 A from partner Arg92: a shorter acid may
                # recover a polar interaction. Discount, do not assert, that rescue.
                polar+=.40 if alt=='K' else (1.15 if alt in 'NQ' else 1.8)
                if alt in 'DE':
                    cg=np.array(coords['C',p]['atoms']['CG']['coord'])
                    positive=[np.linalg.norm(cg-a['coord']) for r in coords.values()
                              if r['chain']=='A' and r['WT'] in 'RK'
                              for name,a in r['atoms'].items() if name in ['NZ','NH1','NH2']]
                    assert min(positive)<3.5
                    polar-=1.4
            if wt in 'DE' and p!=15:
                # Weak exposed hydrophobe cost; no invented salt bridge.
                if alt in 'ILVFWY':polar+=.15*max(0,HYDRO[alt]-HYDRO[wt])
            # Use exact supplied marginal LR; bound its influence because tiny
            # pseudocount-supported frequencies and unknown Neff are not affinity.
            lr=t['sum_log_frequency_ratio']
            assert lr is not None and t['non_gap_depth']>0
            evolutionary=.15*max(-6.,min(6.,lr))
            score=evolutionary-helix-packing-polar
            terms.append({'position':p,'WT':wt,'ALT':alt,'evolutionary_prior':evolutionary,
                'helix_cost':helix,'packing_cost':packing,'polar_cost':polar,'raw_score':score})
        raw_score=sum(t['raw_score'] for t in terms)
        predictions.append({'id':q['id'],'score':round(math.tanh(raw_score/7.),8)})
        contributions.append({'id':q['id'],'mutation':q['mutation'],'raw_score':raw_score,'terms':terms})
    result={'predictions':predictions,'summary':'Unsupervised additive BH3–Mcl-1 binding prior combines bounded supplied MSA log-frequency ratios with WT-interface burial/pocket space, hydrophobicity, side-chain volume, polar-contact retention and helix propensity. Mcl-1 chain A remains fixed. Public-source hashes, sequence/residue mapping and complete nonzero-depth MSA coverage at every mutated site verified; zero-depth flanks are excluded. One global tanh scale; higher is better. No assay calibration or validated epistasis; unpaired conservation, static modeled WT geometry and possible mutant polar contacts leave substantial ranking uncertainty.'}
    ids=[q['id'] for q in case['query']]
    assert [p['id'] for p in predictions]==ids
    assert set(ids)=={f'M{i:03d}' for i in range(1,251)}
    assert all(math.isfinite(p['score']) and -1<=p['score']<=1 for p in predictions)
    Path('predictions.json').write_text(json.dumps(result,indent=2)+'\n')
    Path('model_diagnostics.json').write_text(json.dumps({'WT_site_features':features,'contributions':contributions},indent=2)+'\n')
    return result


if __name__=='__main__':
    case,msa,structure=load_inputs()
    if not Path('checkpoint_predictions.json').exists():checkpoint(case,msa)
    result=refined(case,msa,structure)
    print('Final artifact saved: 250 verified finite predictions on a global scale.')
