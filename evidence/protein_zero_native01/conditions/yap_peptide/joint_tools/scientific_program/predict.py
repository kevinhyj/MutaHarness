"""Label-free relative binding predictor; checkpoint baseline plus documented refinement."""
import json, math, re, hashlib
from pathlib import Path
import numpy as np
import msa_tools

ROOT=Path(__file__).resolve().parent

def read(name): return json.loads((ROOT/name).read_text())

def validate(case, result):
    refs=case['reference_chains']
    expected=[f'M{i:03d}' for i in range(1,251)]
    assert [q['id'] for q in case['query']]==expected
    assert len(result['predictions'])==250
    assert [p['id'] for p in result['predictions']]==expected
    assert all(math.isfinite(p['score']) and -1<=p['score']<=1 for p in result['predictions'])
    assert case['context']['higher_is_better'] is True
    for q in case['query']:
        changes=msa_tools._candidate_substitutions(q,refs)
        declared=[]
        for token in q['mutation'].split(';'):
            m=re.fullmatch(r'([^:]+):([A-Z])(\d+)([A-Z])',token)
            assert m is not None
            chain,wt,pos,alt=m.groups(); pos=int(pos)
            assert refs[chain][pos-1]==wt
            if wt!=alt: declared.append((chain,pos,wt,alt))
        assert sorted(declared)==sorted((s['chain'],s['position'],s['WT'],s['ALT']) for s in changes)
        assert q['sequence']==q['mutated_chains'][case['context']['variant_chain']]
        for chain,seq in case['context']['fixed_partner_chains'].items(): assert q['mutated_chains'][chain]==seq


def checkpoint():
    case=read('case.json'); supplied=read('msa_evidence.json')
    computed=msa_tools.score_case(case)
    assert computed['reference_sequences_sha256']==supplied['reference_sequences_sha256']
    vals=[]
    for a,b in zip(computed['candidates'],supplied['candidates']):
        assert a['id']==b['id']
        score=a['scores']['sum_log_frequency_ratio']
        assert score==b['scores']['sum_log_frequency_ratio']
        assert score['complete'], 'Do not use partial sums as complete evolutionary evidence'
        vals.append(score['score'])
    scores=np.tanh(np.array(vals)/10)
    result={'predictions':[{'id':q['id'],'score':round(float(v),8)} for q,v in zip(case['query'],scores)],'summary':'Unsupervised checkpoint: complete supplied independent-site homolog log-frequency ratios relative to WT, summed across substitutions and mapped by one global tanh(score/10). Higher favors evolutionary compatibility; fixed peptide P retained and mutation identities checked. No measured binding outcomes; natural conservation is an imperfect binding prior, with unknown epistasis and homolog effective depth.'}
    validate(case,result)
    (ROOT/'checkpoint_predictions.json').write_text(json.dumps(result,indent=2)+'\n')
    return case, computed

if __name__=='__main__':
    case,msa=checkpoint()
    print('Saved and validated 250 checkpoint predictions; complete MSA coverage for every candidate.')

# The supplemental raw profile is kept distinct from the unchanged supplied
# frequencies and the fixed msa_tools scores. No assay labels enter this model.
AA='ACDEFGHIKLMNPQRSTVWY'
VOLUME=dict(zip(AA,[88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.0,116.1,140.0,227.8,193.6]))
HYDROPHOBIC=set('AVILMFWY')
POLAR=set('STNQDEHKRY')
CHARGED=set('DEKR')
AROMATIC=set('FWY')

def binding_class_profile(case):
    from scipy.spatial.distance import cdist
    paths=[p for p in (ROOT/'raw_evidence').glob('*.a3m')]
    assert len(paths)==1
    sequences=[]; current=''
    for line in paths[0].read_text().splitlines():
        if line.startswith('>'):
            if current: sequences.append(re.sub(r'[a-z.]','',current))
            current=''
        else: current+=line.strip()
    if current: sequences.append(re.sub(r'[a-z.]','',current))
    wt=case['reference_sequence']
    assert sequences[0]==wt
    assert all(len(s)==len(wt) for s in sequences)
    seqs=list(dict.fromkeys(s for s in sequences[1:] if s!=wt))
    a=np.array([list(s) for s in seqs])
    # Recognition-class anchors: first core Trp, aromatic proline groove,
    # Tyr-recognition His, second Trp. H28 contacts the fixed peptide Tyr OH.
    keep=(a[:,12]=='W') & np.isin(a[:,23],list('FY')) & (a[:,27]=='H') & (a[:,34]=='W') & (np.isin(a[:,8:38],list(AA)).mean(1)>=.9)
    a=a[keep]
    assert len(a)>0
    core=np.array([[ord(v) for v in row] for row in a[:,8:38]],dtype=np.uint8)
    weights=np.empty(len(a))
    for start in range(0,len(a),256):
        d=cdist(core[start:start+256],core,metric='hamming')
        weights[start:start+256]=1/np.maximum(1,(d<=.2+1e-9).sum(1))
    counts=np.array([[(weights*(a[:,p]==v)).sum() for v in AA] for p in range(len(wt))])
    depths=counts.sum(1)
    freqs=(counts+.05)/(depths[:,None]+1)
    # This is neighborhood redundancy weighting, not an independent-sample Neff.
    return freqs, {'raw_rows':len(sequences),'nonquery_unique_rows':len(seqs), 'selected_rows':len(a),'sum_redundancy_weights':float(weights.sum()),'weighted_non_gap_depth_by_position':depths.tolist(),'weighting':'inverse number of selected sequences within 20% Hamming distance on match columns 9..38; gaps count as mismatches','class_anchors_1based':[13,24,28,35],'anchor_policy':'original supplied profile at selection anchors, to avoid artificial conservation caused by conditioning'}

def dihedral(points):
    b=np.array(points,dtype=float)
    b0=-(b[1]-b[0]);b1=b[2]-b[1];b2=b[3]-b[2]
    b1/=np.linalg.norm(b1)
    v=b0-np.dot(b0,b1)*b1;w=b2-np.dot(b2,b1)*b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1,v),w),np.dot(v,w))))

def final_predict(case,msa):
    import structure_tools as st
    manifest=read('evidence_manifest.json');structure=read('structure_evidence.json')
    assert manifest['measurements_present'] is False and manifest['support_count']==0
    hashes={}
    for name,metadata in manifest['files'].items():
        if isinstance(metadata,dict):
            hashes[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            assert hashes[name]==metadata['sha256']
    sequence_hashes={k:hashlib.sha256(v.encode()).hexdigest() for k,v in case['reference_chains'].items()}
    assert msa['reference_sequences_sha256']==sequence_hashes
    raw=structure['raw_coordinate_evidence']
    assert raw['reference_sequence_sha256']==sequence_hashes['A']
    assert raw['raw_source']['sha256']==hashes['raw_evidence/1JMQ_hm.pdb']
    recomputed=st.raw_structure_evidence(case,ROOT/'raw_evidence/1JMQ_hm.pdb')
    for field in ['sites','mapping','heavy_atom_residue_contact_graph','CA_residue_contact_graph']:
        assert recomputed[field]==raw[field], field
    residues,mapping=st._map_raw_residues(st._prepare(case),st._read_pdb(ROOT/'raw_evidence/1JMQ_hm.pdb',include_zero_occupancy=raw['raw_source']['zero_occupancy_atoms_included']),None)
    atoms={(r['chain'],r['position']):r['atoms'] for r in residues if 'position' in r}
    sites={(s['chain'],s['position']):s for s in raw['sites']}
    angles={}
    for (ch,p),a in atoms.items():
        if (ch,p-1) in atoms and (ch,p+1) in atoms:
            prev,nxt=atoms[ch,p-1],atoms[ch,p+1]
            phi=dihedral([prev['C']['coord'],a['N']['coord'],a['CA']['coord'],a['C']['coord']])
            psi=dihedral([a['N']['coord'],a['CA']['coord'],a['C']['coord'],nxt['N']['coord']])
            angles[ch,p]=(phi,psi)
    supplemental,profile_diag=binding_class_profile(case)
    allterms=[];latent=[]
    for q,ev in zip(case['query'],msa['candidates']):
        assert q['id']==ev['id']
        terms=[]
        for sub in ev['substitution_terms']:
            ch,p,wt,alt=(sub[k] for k in ['chain','position','WT','ALT'])
            assert ch=='A' and case['reference_chains'][ch][p-1]==wt
            lr=sub['sum_log_frequency_ratio']
            # Null or zero-depth evidence would require an explicitly separate
            # physical fallback, not using the supplied partial sum. None occur.
            assert lr is not None and sub['non_gap_depth']>0
            raw_lr=math.log(supplemental[p-1,AA.index(alt)]/supplemental[p-1,AA.index(wt)])
            evo=lr if p in [13,24,28,35] else .7*lr+.3*raw_lr
            s=sites[ch,p]
            assert s['WT']==wt and s['coordinate_present'] and s['sidechain_complete'] and s['backbone_complete']
            # RSA here is reproduced raw-coordinate RSA, rather than missing
            # canonical RSA; bound only for defining this heuristic's weight.
            rsa=float(s['isolated_chain_wt_RSA']);exposure=min(1,max(0,rsa));burial=1-exposure
            interface=min(1,max(0,s['observed_atom_SASA_buried_by_context_A2']/40))
            reliability=.55+.45*max(burial,interface)
            if p<9 or p>38: reliability*=.75
            evolution=reliability*evo
            # Small physical corrections identify chemistry absent from a
            # frequency ratio. These are dimensionless priors, never energies.
            physical=0.
            if wt in HYDROPHOBIC and alt not in HYDROPHOBIC:
                physical-=.9*burial
            if alt in CHARGED and wt not in CHARGED:
                physical-=.8*burial
            physical-=.7*burial*max(0,VOLUME[wt]-VOLUME[alt])/100
            physical-=.4*burial*max(0,VOLUME[alt]-VOLUME[wt]-35)/100
            if wt not in HYDROPHOBIC and alt in HYDROPHOBIC:
                physical-=.2*exposure*(1-interface)
            if wt in AROMATIC and alt not in AROMATIC:
                physical-=.7*max(burial,interface)
            if wt in HYDROPHOBIC and alt not in HYDROPHOBIC:
                physical-=.6*interface
            if wt in POLAR and alt not in POLAR:
                physical-=.45*interface
            phi,psi=angles[ch,p]
            if wt=='G' and phi>0:
                physical-=1.0 if alt in set('ADNSC') else 2.5
            if alt=='P':
                if phi>0: physical-=3.0
                elif phi<-100: physical-=1.5
                # Loss of a WT nonlocal backbone N-H hydrogen-bond geometry.
                n=np.array(atoms[ch,p]['N']['coord'])
                hbond=any(np.linalg.norm(n-np.array(other['O']['coord']))<3.4 for (och,op),other in atoms.items() if och==ch and abs(op-p)>2 and 'O' in other)
                if hbond: physical-=.8
            if phi>0 and wt!='G' and alt in set('IVT'): physical-=.8
            score=evolution+physical
            terms.append({'chain':ch,'position':p,'WT':wt,'ALT':alt,'original_log_frequency_ratio':lr,'supplemental_log_frequency_ratio':raw_lr,'evolution_weight':reliability,'evolution_contribution':evolution,'physical_contribution':physical,'score':score,'wt_phi_deg':phi,'wt_psi_deg':psi,'wt_isolated_RSA':rsa,'fixed_partner_burial_A2':s['observed_atom_SASA_buried_by_context_A2']})
        latent.append(math.fsum(t['score'] for t in terms))
        allterms.append({'id':q['id'],'mutation':q['mutation'],'terms':terms,'latent_score':latent[-1]})
    # One monotonic mapping for the entire panel. Zero denotes panel median,
    # not measured WT-like or neutral performance.
    center=float(np.median(latent));scale=6.
    transformed=np.tanh((np.array(latent)-center)/scale)
    summary=('Label-free additive ranking: supplied complete homolog log-frequency ratios, modestly blended with a PPXY-class, redundancy-weighted profile from the verified raw MSA; WT exposure, backbone constraints and contacts with fixed peptide P supply small physical corrections. Raw hashes, exact chain/residue mapping, all mutations and complete evidence coverage verified. One global tanh scale centered on the panel median; higher predicts better binding. No assay labels: conservation and static WT geometry are imperfect priors, scores are uncalibrated, and mutant relaxation, partner coupling and epistasis remain unknown.')
    result={'predictions':[{'id':q['id'],'score':round(float(v),8)} for q,v in zip(case['query'],transformed)],'summary':summary}
    validate(case,result)
    (ROOT/'final_predictions.json').write_text(json.dumps(result,indent=2)+'\n')
    audit={'source_hashes':hashes,'reference_sequences_sha256':sequence_hashes,'mapping':mapping,'coverage':{'candidate_count':len(latent),'complete_msa_candidate_count':sum(ev['scores']['sum_log_frequency_ratio']['complete'] for ev in msa['candidates']),'mapped_residues':len(sites),'missing_heavy_atoms':sum(bool(s['missing_standard_heavy_atoms']) for s in sites.values()),'original_profile_neff':msa['profile_neff']},'supplemental_profile':profile_diag,'global_transform':{'formula':'tanh((latent-panel_median)/6)','panel_median':center,'scale':scale},'candidates':allterms}
    (ROOT/'prediction_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('Saved validated final_predictions.json and prediction_audit.json.')
    print('Top 12:',[(q['id'],q['mutation'],round(v,3)) for q,v in sorted(zip(case['query'],transformed),key=lambda x:-x[1])[:12]])
    print('Bottom 8:',[(q['id'],q['mutation'],round(v,3)) for q,v in sorted(zip(case['query'],transformed),key=lambda x:x[1])[:8]])
    return result

if __name__=='__main__':
    final_predict(case,msa)
