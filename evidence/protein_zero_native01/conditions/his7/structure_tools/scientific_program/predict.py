"""Zero-shot HIS3 ranking. No outcomes or fitted coefficients are used.
Initial checkpoint: conservative substitution and static-WT structural disruption.
"""
import json, math, re, hashlib
from pathlib import Path
import numpy as np

HYDRO = dict(zip('ACDEFGHIKLMNPQRSTVWY',[1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
VOLUME = dict(zip('ACDEFGHIKLMNPQRSTVWY',[88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.0,116.1,140.0,227.8,193.6]))
CHARGE={a:float(a in 'KR')-float(a in 'DE') for a in HYDRO}
GROUPS=['ILVM','FY','ST','NQ','DE','KR','AG','C','H','P','W']

def validate(case, packet, manifest):
    wt=case['reference_sequence']; assert len(case['query'])==250
    assert manifest['support_count']==0 and not manifest['measurements_present']
    assert case['context']['higher_is_better']
    assert hashlib.sha256(wt.encode()).hexdigest()==packet['reference_sequence_sha256']
    declared=manifest['files']['raw_evidence/HIS7_YEAST.pdb']['sha256']
    actual=hashlib.sha256(Path('raw_evidence/HIS7_YEAST.pdb').read_bytes()).hexdigest()
    assert actual==declared==packet['raw_coordinate_evidence']['raw_source']['sha256']
    assert packet['raw_coordinate_evidence']['mapping'][0]['method']=='exact_whole_chain_sequence'
    ids=[]
    for q in case['query']:
        ids.append(q['id']); seq=list(wt); seen=set()
        for mut in q['mutation'].split(':'):
            assert re.fullmatch('[A-Z][0-9]+[A-Z]',mut)
            p=int(mut[1:-1]); assert p not in seen and wt[p-1]==mut[0]
            seen.add(p);seq[p-1]=mut[-1]
        assert ''.join(seq)==q['sequence'] and len(seen)==q['mutation_order']
    assert len(set(ids))==250 and set(ids)=={f'M{i:03d}' for i in range(1,251)}
    assert not case.get('reference_chains') and not case['context'].get('fixed_partners')
    return {'reference_sha256':packet['reference_sequence_sha256'],'pdb_sha256':actual,
            'mapping':'Exact whole-chain sequence; all 220 residues observed.',
            'fixed_partners':'None supplied; single-chain coordinates do not represent IGPD biological oligomer or metal cofactors.',
            'msa':'Absent; not treated as zero evolutionary fitness.'}

def initial_site_cost(w,a,rsa,phi,psi):
    if w==a:return 0.
    exposure=min(1.,max(0.,rsa)) if rsa is not None else None
    burial=max(0.,1.-exposure/0.5) if exposure is not None else 0.5
    conservative=any(w in g and a in g for g in GROUPS)
    cost=0.05+(0.04 if conservative else 0.14)
    cost+=burial*0.35*abs(VOLUME[a]-VOLUME[w])/60
    if w in 'AVILMFCWY' and a in 'DEKR':cost+=1.5*burial
    if w in 'AVILMFCWY' and a in 'NQSTH':cost+=0.40*burial
    if CHARGE[w]*CHARGE[a]<0:cost+=0.3+0.5*burial
    if w in 'DEKR' and CHARGE[a]==0:cost+=0.15+0.15*burial
    if a=='G' and w!='G':cost+=0.18+0.30*burial
    if a=='P' and w!='P':
        cost+=0.20
        if phi is not None and (phi < -95 or phi>0):cost+=1.25
    if w=='P' and a!='P':cost+=0.30
    if w=='G' and a!='G' and phi is not None and phi>0:cost+=0.8
    return cost

def run():
    case=json.load(open('case.json')); packet=json.load(open('structure_evidence.json')); manifest=json.load(open('evidence_manifest.json'))
    audit=validate(case,packet,manifest)
    raw={s['position']:s for s in packet['raw_coordinate_evidence']['sites']}
    summary={s['position']:s['wt_structure'] for s in case['evidence']['sites']}
    details=[]; predictions=[]
    for q in case['query']:
        costs=[]
        for mut in q['mutation'].split(':'):
            p=int(mut[1:-1]); s=summary[p]; r=raw[p]
            cost=initial_site_cost(mut[0],mut[-1],r.get('isolated_chain_wt_RSA'),s.get('phi_deg'),s.get('psi_deg'))
            costs.append((mut,cost))
        risk=sum(x[1] for x in costs)
        score=math.tanh((1.5-risk)/2.5)
        predictions.append({'id':q['id'],'score':round(score,8)})
        details.append({'id':q['id'],'risk':risk,'site_costs':costs})
    result={'predictions':predictions,'summary':'Unsupervised structural-disruption prior for HIS3 growth: conservative substitutions, WT burial, side-chain chemistry and glycine/proline backbone compatibility. One global monotonic scale; higher means predicted better growth. Mutation identities and all IDs checked; reference/PDB hashes match and coordinate mapping is exact. No MSA or measured outcomes; static single-chain WT geometry omits oligomer/cofactor effects and cannot validate epistasis. Rankings have substantial uncertainty.'}
    assert all(math.isfinite(x['score']) and -1<=x['score']<=1 for x in predictions)
    Path('checkpoint_predictions.json').write_text(json.dumps(result,indent=2)+'\n')
    Path('checkpoint_diagnostics.json').write_text(json.dumps({'audit':audit,'candidates':details},indent=2)+'\n')
    print('Saved checkpoint: 250 finite scores; range',min(x['score'] for x in predictions),max(x['score'] for x in predictions))

if __name__=='__main__':run()
