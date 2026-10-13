"""Unsupervised sequence/fold-prior stability ranking; no measured outcomes.
All position assignments are hypotheses from chromodomain fold knowledge,
not observations of this construct's geometry. Scores use one global map.
"""
import json, math, re
from pathlib import Path
CASE=json.loads(Path('case.json').read_text())
WT=CASE['reference_sequence']
# Approximate burial prior (0=exposed, 1=buried), not solvent accessibility data.
BURIAL={5:.65,8:.25,9:.55,10:.65,15:.65,16:.45,17:.55,18:.85,19:.8,
        21:.65,24:.65,28:.6,29:.25,30:.7,31:.7,36:.55,37:.65,38:.75,
        42:.65,43:.8,44:.65,45:.25,46:.9}
HYDRO=dict(zip('ACDEFGHIKLMNPQRSTVWY',[1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]))
# Fauchere-Pliska side-chain transfer scale handles aromatic hydrophobicity
# more appropriately than the original Kyte-Doolittle checkpoint feature.
TRANSFER=dict(zip('ACDEFGHIKLMNPQRSTVWY',[.31,1.54,-.77,-.64,1.79,0,.13,1.8,-.99,1.7,1.23,-.6,.72,-.22,-1.01,-.04,.26,1.22,2.25,.96]))
VOL=dict(zip('ACDEFGHIKLMNPQRSTVWY',[88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.,116.1,140.,227.8,193.6]))
CHARGE={a:(1 if a in 'KR' else -1 if a in 'DE' else .2 if a=='H' else 0) for a in HYDRO}
BETA=set(range(5,13))|set(range(15,23))|set(range(28,32))
HELIX=set(range(41,49))

def validate():
    ids=[]
    for q in CASE['query']:
        ids.append(q['id']); seq=list(WT); seen=set()
        muts=q['mutation'].split(':')
        assert len(muts)==q['mutation_order']
        for m in muts:
            match=re.fullmatch(r'([A-Z])(\d+)([A-Z])',m); assert match,m
            a,p,b=match.groups(); p=int(p)
            assert p not in seen and 1<=p<=len(WT) and WT[p-1]==a and a!=b,m
            assert b in HYDRO,m
            seen.add(p);seq[p-1]=b
        assert ''.join(seq)==q['sequence'],q['id']
    assert len(ids)==250 and set(ids)=={f'M{i:03d}' for i in range(1,251)}
    assert len(ids)==len(set(ids))
    assert CASE['context']['higher_is_better'] is True
    assert not any(k in CASE for k in ['partners','fixed_partners']), 'Inspect supplied partners'

def single(a,p,b):
    bur=BURIAL.get(p,.05)
    # Coarse physical terms; arbitrary energy units, not calibrated kcal/mol.
    dh=TRANSFER[b]-TRANSFER[a]
    e=.8*bur*max(0,-dh)-.6*bur*max(0,dh)
    e+=.12*(1-bur)*max(0,dh)
    e+=bur*(.007*max(0,VOL[a]-VOL[b])+.009*max(0,VOL[b]-VOL[a]-10))
    e+=1.3*bur*max(0,abs(CHARGE[b])-abs(CHARGE[a]))
    e+=.07*(1-bur)*abs(CHARGE[b]-CHARGE[a])
    if a in 'FWY' and b not in 'FWY': e+=1.0*bur
    if b=='P' and p in BETA|HELIX: e+=2.0
    if b=='G' and p in BETA|HELIX: e+=.75
    if a=='G': e+=.65+(.35 if VOL[b]>140 else 0)
    if a=='P': e+=.7
    if p in (29,30,36):
        # Partial burial of the YNTW/ENIL region makes replacing polar
        # side chains more costly than hydrophobicity alone would suggest.
        compatible=('ST' if a=='T' else 'NQ')
        if b in compatible: loss=.35
        elif b in 'NQSTH': loss=.65
        elif b in 'DEY': loss=.9
        else: loss=2.2
        e+=bur*loss
    if p==36 and b=='G': e+=.25
    # Small generic substitution cost prevents chemically dissimilar exposed
    # substitutions all being assigned exactly zero; no ID-dependent jitter.
    e+=.035
    if p<=4 or p>=49: e*=.35
    return e

def mutations(q):
    return [(m[0],int(m[1:-1]),m[-1]) for m in q['mutation'].split(':')]

def score(q):
    ms=mutations(q)
    e=sum(single(*m) for m in ms)
    changes={p:b for a,p,b in ms}
    if 9 in changes or 16 in changes:
        # Hypothesized beta-hairpin ion pair. Only this chemically explicit
        # interaction is modeled; other double effects stay additive.
        a,b=changes.get(9,WT[8]),changes.get(16,WT[15])
        ca,cb=CHARGE[a],CHARGE[b]
        if ca*cb < -.5:
            e+=.4 if ca<0 else (.15 if a!='R' or b!='E' else 0)
        else:
            e+=.9
            if ca*cb>.5: e+=1.2
            elif abs(ca)>.5 or abs(cb)>.5: e+=.7
            elif ca or cb: e+=.15
    return math.tanh(-e/3)

SUMMARY=('Unsupervised sequence-based chromodomain stability prior: approximate burial, side-chain '
         'hydrophobicity/volume, aromatic retention, backbone and polar-network compatibility. '
         'Double effects are additive except a speculative R9/E16 charge-pair term. All scores use '
         'one global tanh scale; higher predicts greater proteolysis-assay stability, not calibrated assay units. '
         'All substitutions match the reference and candidate sequences; no fixed partners are supplied. '
         'No outcomes, MSA or coordinates are available (hence no source hashes or structural mapping); '
         'position priors and epistasis have substantial uncertainty.')

if __name__=='__main__':
    validate()
    output={'predictions':[{'id':q['id'],'score':round(score(q),8)} for q in CASE['query']], 'summary':SUMMARY}
    assert all(math.isfinite(x['score']) and -1<=x['score']<=1 for x in output['predictions'])
    Path('checkpoint_predictions.json').write_text(json.dumps(output,indent=2)+'\n')
    Path('final_predictions.json').write_text(json.dumps(output,indent=2)+'\n')
    print('Saved complete validated checkpoint:',len(output['predictions']),'candidates')
