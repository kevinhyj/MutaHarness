"""Label-free YAP WW-domain ranking; no assay outcomes are available."""
import hashlib,json,math,re
from pathlib import Path
import msa_tools

def load_and_validate():
    case=json.loads(Path('case.json').read_text())
    manifest=json.loads(Path('evidence_manifest.json').read_text())
    ev=json.loads(Path('msa_evidence.json').read_text())
    assert manifest['support_count']==0 and manifest['measurements_present'] is False
    assert case['context']['higher_is_better'] is True
    refs=case['reference_chains']
    assert ev['reference_sequences_sha256']=={k:hashlib.sha256(v.encode()).hexdigest() for k,v in refs.items()}
    for fn,meta in manifest['files'].items():
        if isinstance(meta,dict) and 'sha256' in meta:
            assert hashlib.sha256(Path(fn).read_bytes()).hexdigest()==meta['sha256']
    # Bind the supplied profile to raw match columns and its declared cap.
    raw_files=[fn for fn,meta in manifest['files'].items()
               if isinstance(meta,dict) and meta.get('role')=='msa']
    assert len(raw_files)==1
    rows=[];current=None
    for line in Path(raw_files[0]).read_text().splitlines():
        if line.startswith('>'):
            if current is not None:rows.append(current)
            current=''
        else:
            assert current is not None
            current+=''.join(a for a in line.strip() if not a.islower() and a!='.')
    if current is not None:rows.append(current)
    assert rows[0]==refs['A'] and all(len(r)==len(refs['A']) for r in rows)
    unique=[];seen_sequences=set()
    for r in rows[1:]:
        if r==refs['A'] or r in seen_sequences:continue
        seen_sequences.add(r);unique.append(r)
        if len(unique)==2048:break
    assert len(unique)==2048
    for site in case['evidence']['sites']:
        assert site['chain']=='A'
        col=site['position']-1;msa=site['msa']
        assert msa['msa_column_1based']==site['position']
        observed=[r[col] for r in unique if r[col] in msa_tools.AA]
        assert len(observed)==msa['non_gap_depth']
        assert abs(msa['gap_fraction']-(1-len(observed)/2048))<1e-12
        for aa,f in msa['frequencies'].items():
            assert math.isclose(f,(observed.count(aa)+0.05)/(len(observed)+1),abs_tol=1e-12)
    assert len(case['query'])==250
    assert {x['id'] for x in case['query']}=={f'M{i:03d}' for i in range(1,251)}
    for q in case['query']:
        reconstructed={k:list(v) for k,v in refs.items()};seen=set()
        for mutation in q['mutation'].split(';'):
            m=re.fullmatch(r'([A-Z]):([A-Z])(\d+)([A-Z])',mutation);assert m
            chain,wt,p,alt=m.groups();p=int(p)
            assert refs[chain][p-1]==wt and (chain,p) not in seen
            seen.add((chain,p));reconstructed[chain][p-1]=alt
        assert {k:''.join(v) for k,v in reconstructed.items()}==q['mutated_chains']
        assert q['sequence']==q['mutated_chains']['A']
        assert q['mutation_order']==len(seen)
        for k,v in case['context']['fixed_partner_chains'].items():assert q['mutated_chains'][k]==v
    recomputed=msa_tools.score_case(case)
    assert recomputed==ev
    return case,manifest,ev

def save_output(predictions,summary,path):
    assert len(predictions)==250 and len({p['id'] for p in predictions})==250
    assert all(math.isfinite(p['score']) and -1<=p['score']<=1 for p in predictions)
    out={'predictions':predictions,'summary':summary}
    Path(path).write_text(json.dumps(out,indent=2)+'\n')
    return out

# Fixed biochemical priors: log-like relative units, with no fitted outcomes.
# Classification derives from WW-domain sequence/known fold; not measured RSA.
CHEMICAL_GROUPS=[set('ILVM'),set('FY'),set('KR'),set('DE'),set('NQ'),set('ST')]
AROMATIC=set('FWY')
ALIPHATIC=set('AILVM')
STRAND_POSITIONS=set(range(13,17))|set(range(23,28))|set(range(32,36))
CORE_HYDROPHOBES={9,13,15,24,25,26,35}


def chemistry_prior(wt,alt,p):
    conservative=any(wt in g and alt in g for g in CHEMICAL_GROUPS)
    # A mild substitution prior, not a measured neutral-fitness claim.
    value=-0.15 if conservative else -0.45
    if p in CORE_HYDROPHOBES:
        if wt in AROMATIC:
            value=-1.0 if alt in AROMATIC else (-2.3 if alt in ALIPHATIC else -4.5)
            if p in {13,35} and alt not in AROMATIC:value-=1.0
        else:
            value=-0.2 if alt in ALIPHATIC else (-1.2 if alt in AROMATIC else -3.5)
    if wt=='P':value=-2.8 if p in {10,38} else -0.9
    if wt=='G' and p in {12,21}:
        value=-1.5 if alt in 'AS' else -3.0
    if alt=='P' and p in STRAND_POSITIONS:value-=2.5
    if p==27:value=-0.25 if alt in 'DS' else (-1.0 if alt in 'HQ' else -2.3)
    if p==28:value=-1.8 if alt in 'YQR' else -2.8
    return value


def function_prior(p,alt):
    # Qualitative YAP-WW stability/recognition priors, no observed assay values.
    if p==6:
        if alt in 'NAGVQ':return 0.5
        if alt in 'RH':return 0.7
        if alt=='Y':return -0.2
    if p==19 and alt=='R':return 2.0
    if p==30:
        return {'N':0.9,'G':1.6,'A':1.0,'H':0.5,'Q':0.35,
                'V':-0.4,'L':-0.6,'Y':-0.6}.get(alt,0.0)
    if p==31:return {'R':0.6,'K':0.35,'H':0.25,'P':-0.6}.get(alt,0.0)
    if p==36 and alt in 'HRK':return 0.8
    return 0.0


def hybrid_score(candidate):
    terms=[]
    for t in candidate['substitution_terms']:
        assert not t['missing_flags'] and t['non_gap_depth']>0
        assert t['chain']=='A'
        p,wt,alt=t['position'],t['WT'],t['ALT']
        lr=t['sum_log_frequency_ratio']
        assert lr is not None and math.isfinite(lr)
        # Bound zero-count severity; preserve unmodified fixed-score evidence.
        bounded=max(-8.0,min(2.5,lr))
        edge_weight={6:0.25,7:0.65,8:0.75,39:0.40}.get(p,1.0)
        conservative=any(wt in g and alt in g for g in CHEMICAL_GROUPS)
        if conservative:bounded=max(bounded,-2.5)
        chemistry=chemistry_prior(wt,alt,p)
        functional=function_prior(p,alt)
        effect=0.75*edge_weight*bounded+0.25*chemistry+functional
        terms.append({'position':p,'WT':wt,'ALT':alt,'supplied_LR':lr,
                      'bounded_LR':bounded,'edge_weight':edge_weight,
                      'chemistry_prior':chemistry,'function_prior':functional,
                      'latent_effect':effect})
    return math.fsum(t['latent_effect'] for t in terms),terms


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',action='store_true')
    args=parser.parse_args()
    case,manifest,ev=load_and_validate()
    if args.checkpoint:
        predictions=[]
        for q in ev['candidates']:
            s=q['scores']['sum_log_frequency_ratio']
            assert s['complete'] and s['score'] is not None
            predictions.append({'id':q['id'],'score':math.tanh(s['score']/15)})
        summary=('Initial unsupervised checkpoint: complete independent-site natural-homolog log-frequency ratios, globally mapped with tanh(score/15). Mutation identities, fixed peptide GTPPPPYTVG, source and sequence hashes, and all 250 IDs verified. No structure or outcomes; unweighted homolog counts are not Neff. Evolutionary compatibility is an uncertain binding prior; epistasis is unknown.')
        save_output(predictions,summary,'checkpoint_predictions.json')
        return
    predictions=[];audit=[]
    for q in ev['candidates']:
        assert q['scores']['sum_log_frequency_ratio']['complete']
        latent,terms=hybrid_score(q)
        score=math.tanh((latent+5.0)/8.0)
        predictions.append({'id':q['id'],'score':round(score,8)})
        audit.append({'id':q['id'],'mutation':q['mutation'],'latent':latent,'score':score,'terms':terms})
    summary=('Unsupervised YAP WW-domain ranking combining complete homolog log-frequency ratios with modest biochemical priors for core integrity, turns and peptide recognition; flexible boundaries and conservative substitutions receive reduced penalties. One global monotone scale; higher is better. All 250 mutation identities, source/sequence hashes, residue mapping and fixed peptide GTPPPPYTVG verified. No missing evolutionary terms or structures; unweighted depth is not Neff. Scores are uncalibrated: conservation imperfectly predicts binding, and affinity/stability tradeoffs and multi-mutant epistasis remain uncertain without assay outcomes.')
    save_output(predictions,summary,'final_predictions.json')
    Path('prediction_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('Saved final predictions:',len(predictions),'score range',min(p['score'] for p in predictions),max(p['score'] for p in predictions))
    lookup={x['id']:x for x in case['query']}
    for p in sorted(predictions,key=lambda p:p['score'],reverse=True)[:15]:
        print(p['id'],p['score'],lookup[p['id']]['mutation'])

if __name__=='__main__':main()
