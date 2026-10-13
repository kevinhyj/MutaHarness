"""Label-free HIS3 growth ranking. Initial checkpoint; refinement rationale recorded separately."""
import json, hashlib, math
from pathlib import Path
import numpy as np

def load_validate():
    case=json.load(open('case.json')); manifest=json.load(open('evidence_manifest.json'))
    msa=json.load(open('msa_evidence.json')); structure=json.load(open('structure_evidence.json'))
    assert manifest['support_count']==0 and manifest['measurements_present'] is False
    ref=case['reference_sequence']; h=hashlib.sha256(ref.encode()).hexdigest()
    assert msa['reference_sequences_sha256']=={'__single__':h}
    assert structure['reference_sequence_sha256']==h
    for path,desc in manifest['files'].items():
        if isinstance(desc,dict) and 'sha256' in desc:
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==desc['sha256'],path
    assert not case.get('reference_chains') and not case.get('fixed_partners')
    assert case['context']['higher_is_better'] is True
    ids=[q['id'] for q in case['query']]
    assert len(ids)==250 and len(set(ids))==250
    assert set(ids)=={f'M{i:03}' for i in range(1,251)}
    for q in case['query']:
        seq=list(ref); changes=[]
        for mut in q['mutation'].split(':'):
            wt,alt=mut[0],mut[-1];pos=int(mut[1:-1]);assert ref[pos-1]==wt and wt!=alt
            assert pos not in changes;changes.append(pos);seq[pos-1]=alt
        assert ''.join(seq)==q['sequence'] and len(changes)==q['mutation_order']
    assert set(ids)=={q['id'] for q in msa['candidates']}=={q['id'] for q in structure['candidates']}
    return case,msa,structure

def initial_checkpoint(case,msa):
    indexed={q['id']:q for q in msa['candidates']}; rows=[];missing=0
    for q in case['query']:
        terms=indexed[q['id']]['substitution_terms'];latent=0.
        for t in terms:
            value=t['sum_log_frequency_ratio']
            if value is None or not t.get('non_gap_depth',0):
                missing+=1;latent-=0.4
            else:latent+=float(value)
        rows.append({'id':q['id'],'score':float(np.tanh(latent/14.))})
    result={'predictions':rows,'summary':'Initial unsupervised checkpoint: supplied natural-homolog log-frequency ratios, with a weak mutation burden where homolog evidence is absent; one global tanh scale, higher better. Source hashes and mutation identities verified; no assay labels or fixed partners supplied. Partial evidence is not a complete evolutionary score; growth and epistasis remain uncertain.'}
    validate_result(result,case);json.dump(result,open('checkpoint_predictions.json','w'),indent=2)
    print('Checkpoint written:',len(rows),'scores; missing mutation terms',missing,'range',min(r['score'] for r in rows),max(r['score'] for r in rows))

def validate_result(result,case):
    assert len(result['predictions'])==250
    assert {r['id'] for r in result['predictions']}=={q['id'] for q in case['query']}
    assert all(math.isfinite(r['score']) and -1<=r['score']<=1 for r in result['predictions'])


# All constants below are fixed prior assumptions, never fitted to assay outcomes.
AA='ACDEFGHIKLMNPQRSTVWY'
AAMAP={a:i for i,a in enumerate(AA)}
# Approximate side-chain/residue volume and Kyte-Doolittle hydropathy.
VOLUME=dict(zip(AA,[88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.,116.1,140.,227.8,193.6]))
HYDRO=dict(zip(AA,[1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]))
CHARGE={a:float(a in 'KR')-float(a in 'DE') for a in AA}

def chemical_distance_table():
    groups=['AVLIM','FWY','STNQ','KR','DE','H','G','P','C']
    group={a:i for i,g in enumerate(groups) for a in g}
    d=np.ones((21,21))
    for a in AA:
        for b in AA:
            d[AAMAP[a],AAMAP[b]]=0 if a==b else (.35 if group[a]==group[b] else 1.)
    # Small polar and histidine exchanges should not be treated as maximally distant.
    for g in ['CST','HNQ','HY']:
        for a in g:
            for b in g:
                if a!=b:d[AAMAP[a],AAMAP[b]]=min(d[AAMAP[a],AAMAP[b]],.55)
    return d

class ContactContext:
    """A fixed, shrunk natural-co-occurrence descriptor on observed WT contacts.

    No mutant coordinates, learned couplings, phenotype labels or validated
    epistasis are used. Omitted oligomeric/ligand contacts remain unmodelled.
    """
    def __init__(self,case,structure):
        z=np.load('homolog_prior.npz');self.aln=z['alignment'];self.weight=z['weights'];self.freq=z['frequencies'];self.depth=z['depth']
        self.wt=np.array([AAMAP[a] for a in case['reference_sequence']]);self.dist=chemical_distance_table()
        graph=structure['raw_coordinate_evidence']['CA_residue_contact_graph']
        self.neighbors=[set() for _ in self.wt]
        for e in graph:
            i,j=e['left']['position']-1,e['right']['position']-1
            if abs(i-j)<=2:continue
            if self.depth[i]<=0 or self.depth[j]<=0:continue
            self.neighbors[i].add(j);self.neighbors[j].add(i)
        self.neighbors=[np.array(sorted(x),dtype=int) for x in self.neighbors]
        self.baseline=np.array([self.probabilities(i,self.wt) for i in range(len(self.wt))])
    def probabilities(self,i,seq):
        neighbors=self.neighbors[i]
        if not len(neighbors):return self.freq[i]
        observed=self.aln[:,neighbors]<20
        shared=observed.sum(1)
        mismatch=(self.dist[self.aln[:,neighbors],seq[neighbors]]*observed).sum(1)
        # Missing homolog context is not a match. Up to half missing is retained
        # at reduced weight, not imputed as a residue.
        coverage=shared/len(neighbors)
        weight=self.weight*np.exp(-.7*mismatch)*coverage*(coverage>=.5)
        valid=self.aln[:,i]<20
        counts=np.bincount(self.aln[valid,i],weights=weight[valid],minlength=20)
        local_depth=weight[valid].sum()
        # Ten effective homolog observations of shrinkage toward the marginal.
        return (counts+10*self.freq[i])/(local_depth+10)
    def correction(self,seq,positions):
        affected=set(positions)
        for i in positions:affected.update(self.neighbors[i].tolist())
        deltas=[]
        for i in affected:
            if self.depth[i]<=0:continue
            if np.array_equal(seq[self.neighbors[i]],self.wt[self.neighbors[i]]):p=self.baseline[i]
            else:p=self.probabilities(i,seq)
            delta=float(np.log(p[seq[i]])-np.log(self.baseline[i,self.wt[i]]))
            marginal=float(np.log(self.freq[i,seq[i]])-np.log(self.freq[i,self.wt[i]]))
            # Conditional deviations are noisy and may reflect shared ancestry.
            deltas.append(float(np.clip(delta-marginal,-2.,2.)))
        return math.fsum(deltas)

def geometry_penalty(wt,alt,observation,raw):
    rsa=raw.get('isolated_chain_wt_RSA')
    confidence=raw.get('source_asserted_AF2_pLDDT')
    if rsa is None:return 0.,False
    exposure=float(np.clip(rsa,0.,1.));burial=float(np.clip((.5-exposure)/.5,0.,1.))
    reliability=1. if confidence is None else float(np.clip((confidence-50)/30,0.,1.))
    # Buried chemistry changes are penalized, without computing mutant energy.
    hydrophobic_loss=max(0.,HYDRO[wt]-HYDRO[alt]-1.)/4.
    penalty=.65*burial*hydrophobic_loss
    if wt in 'AVLIMFWYC' and alt in 'KRDE':penalty+=2.6*burial
    if wt in 'AVLIMFWYC' and alt in 'NQ':penalty+=1.0*burial
    # Cavities and large expansions are weaker priors than introducing charge.
    dv=VOLUME[alt]-VOLUME[wt]
    penalty+=burial*(max(0.,-dv-20)/100 + max(0.,dv-30)/150)
    phi=observation.get('phi_deg');psi=observation.get('psi_deg')
    if alt=='P' and wt!='P':
        penalty+=.3
        if phi is not None:penalty+=max(0.,abs(phi+65)-25)/30
        if phi is not None and psi is not None and -95<phi<-35 and -80<psi<0:penalty+=.6
    if wt=='P' and alt!='P':penalty+=.35
    if wt=='G' and alt!='G' and phi is not None and phi>0:penalty+=2.
    if alt=='G' and wt not in 'GAS':penalty+=.7*burial
    return float(reliability*penalty),True

def refined_predictions(case,msa,structure):
    context=ContactContext(case,structure)
    source={q['id']:q for q in msa['candidates']}
    sites={q['position']:q for q in case['evidence']['sites']}
    raw={q['position']:q for q in structure['raw_coordinate_evidence']['sites']}
    rows=[];diagnostics=[]
    for q in case['query']:
        seq=np.array([AAMAP[a] for a in q['sequence']]);terms=[];penalties=[];positions=[];missing=0
        for term in source[q['id']]['substitution_terms']:
            pos=term['position'];i=pos-1;wt=term['WT'];alt=term['ALT'];positions.append(i)
            broad=term['sum_log_frequency_ratio']
            if broad is None or term.get('non_gap_depth',0)<=0:
                # Explicit prior only: no partial sum is labelled a complete score.
                missing+=1;broad=-.4
            if context.depth[i]>0:
                near=float(np.log(context.freq[i,AAMAP[alt]])-np.log(context.freq[i,AAMAP[wt]]))
                ratio=.45*broad+.55*near
            else:ratio=broad
            penalty,known=geometry_penalty(wt,alt,sites[pos]['wt_structure'],raw[pos])
            terms.append(ratio);penalties.append(penalty)
        # Positive conservation changes have weaker phenotype direction evidence
        # on an already functional natural scaffold; avoid masking a bottleneck.
        additive=math.fsum(min(t,0.)+.3*max(t,0.) for t in terms)
        physical=math.fsum(penalties)
        bottleneck=max(max(0.,-t)+.7*p for t,p in zip(terms,penalties))
        contact_correction=context.correction(seq,positions)
        # Contact pseudo-likelihood counts effects from both residues of an edge;
        # use a small correction, not a full fitted interaction model.
        latent=additive-.7*physical+.25*contact_correction-.3*bottleneck-.12*len(terms)
        score=float(np.tanh(latent/12.))
        rows.append({'id':q['id'],'score':score})
        diagnostics.append({'id':q['id'],'mutation':q['mutation'],'msa_complete':missing==0,'mutation_count':len(terms),'evolutionary_terms':terms,'additive_evolution':additive,'structure_penalty':physical,'natural_contact_correction':contact_correction,'bottleneck':bottleneck,'latent':latent,'score':score})
    result={'predictions':rows,'summary':'Unsupervised growth ranking blends supplied homolog log-frequency ratios with a diversity-weighted yeast-like homolog prior, a strongly shrunk natural contact-context correction, and modest WT burial/backbone penalties. Source hashes, residue mapping and all mutation identities were verified; all mutated sites have homolog and WT-coordinate coverage. Unaligned sites were treated as absent evidence. One global tanh scale; higher is better. No assay outcomes, catalytic partners or oligomeric context were supplied; scores are uncalibrated and epistasis and growth effects remain uncertain.'}
    validate_result(result,case)
    json.dump(result,open('final_predictions.json','w'),indent=2)
    json.dump(diagnostics,open('prediction_diagnostics.json','w'),indent=2)
    print('Final predictions saved:',len(rows),'score range',min(x['score'] for x in rows),max(x['score'] for x in rows))
    return result,diagnostics

if __name__=='__main__':
    case,msa,structure=load_validate()
    initial_checkpoint(case,msa)
    from inspect_alignment import read_alignment
    from build_homolog_prior import build
    records,a,positions=read_alignment()
    ng=(a[1:]<20)&(a[0]<20)
    identity=((a[1:]==a[0])&ng).sum(1)/np.maximum(ng.sum(1),1)
    np.savez_compressed('alignment_parsed.npz',alignment=a,positions=positions,identities=identity,coverage=ng.sum(1)/(a[0]<20).sum())
    build()
    refined_predictions(case,msa,structure)
