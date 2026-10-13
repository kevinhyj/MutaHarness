"""Single principled refinement of predict.py; all weights are fixed heuristics.
Uses generic BLOSUM62 similarity, observed WT chemistry, packing and backbone.
No outcome training, block-specific scoring, candidate-frequency prior or search.
"""
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
from predict import validate, VOLUME, CHARGE, HYDRO

# Standard generic substitution matrix, NOT assay-specific conservation or MSA.
ORDER='ARNDCQEGHILKMFPSTWYV'
ROWS='''
4 -1 -2 -2 0 -1 -1 0 -2 -1 -1 -1 -1 -2 -1 1 0 -3 -2 0
-1 5 0 -2 -3 1 0 -2 0 -3 -2 2 -1 -3 -2 -1 -1 -3 -2 -3
-2 0 6 1 -3 0 0 0 1 -3 -3 0 -2 -3 -2 1 0 -4 -2 -3
-2 -2 1 6 -3 0 2 -1 -1 -3 -4 -1 -3 -3 -1 0 -1 -4 -3 -3
0 -3 -3 -3 9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1 1 0 0 -3 5 2 -2 0 -3 -2 1 0 -3 -1 0 -1 -2 -1 -2
-1 0 0 2 -4 2 5 -2 0 -3 -3 1 -2 -3 -1 0 -1 -3 -2 -2
0 -2 0 -1 -3 -2 -2 6 -2 -4 -4 -2 -3 -3 -2 0 -2 -2 -3 -3
-2 0 1 -1 -3 0 0 -2 8 -3 -3 -1 -2 -1 -2 -1 -2 -2 2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3 4 2 -3 1 0 -3 -2 -1 -3 -1 3
-1 -2 -3 -4 -1 -2 -3 -4 -3 2 4 -2 2 0 -3 -2 -1 -2 -1 1
-1 2 0 -1 -3 1 1 -2 -1 -3 -2 5 -1 -3 -1 0 -1 -3 -2 -2
-1 -1 -2 -3 -1 0 -2 -3 -2 1 2 -1 5 0 -2 -1 -1 -1 -1 1
-2 -3 -3 -3 -2 -3 -3 -3 -1 0 0 -3 0 6 -4 -2 -2 1 3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4 7 -1 -1 -4 -3 -2
1 -1 1 0 -1 0 0 0 -1 -2 -2 0 -1 -2 -1 4 1 -3 -2 -2
0 -1 0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1 1 5 -2 -2 0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1 1 -4 -3 -2 11 2 -3
-2 -2 -2 -3 -2 -1 -2 -3 2 -1 -1 -2 -1 3 -3 -2 -2 2 7 -1
0 -3 -3 -3 -1 -2 -2 -3 -3 3 1 -2 1 -1 -2 -2 0 -3 -1 4
'''
B=np.array([[int(x) for x in row.split()] for row in ROWS.strip().splitlines()]); assert B.shape==(20,20) and np.array_equal(B,B.T)
IDX={a:i for i,a in enumerate(ORDER)}
BACKBONE={'N','CA','C','O','OXT'}
POLAR={'S':{'OG'},'T':{'OG1'},'N':{'OD1','ND2'},'Q':{'OE1','NE2'},'D':{'OD1','OD2'},'E':{'OE1','OE2'},'H':{'ND1','NE2'},'K':{'NZ'},'R':{'NE','NH1','NH2'},'Y':{'OH'},'C':{'SG'},'W':{'NE1'}}

def load_atoms():
    atoms=defaultdict(dict)
    for l in open('raw_evidence/HIS7_YEAST.pdb'):
        if l.startswith('ATOM') and l[21]=='A' and float(l[54:60])>0:
            p=int(l[22:26]); name=l[12:16].strip()
            if l[76:78].strip()!='H':atoms[p][name]=np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])])
    return atoms

def environment(wt,atoms):
    # Weak flags for existing polar contacts. No donor-angle or mutant HB inference.
    polar_contacts={}; donor_contacts={}; close_sc={}; his_clusters=set()
    for p,aa in atoms.items():
        relevant=POLAR.get(wt[p-1],set()); interactions=set(); donor=False
        for q,bb in atoms.items():
            if abs(p-q)<=1:continue
            for a in relevant:
                if a not in aa:continue
                for b,y in bb.items():
                    if b[0] in 'NO' and np.linalg.norm(aa[a]-y)<3.5: interactions.add(q)
            if 'N' in aa and any(b[0]=='O' and np.linalg.norm(aa['N']-y)<3.3 for b,y in bb.items()):donor=True
        polar_contacts[p]=interactions; donor_contacts[p]=donor
    his=[p for p in atoms if wt[p-1]=='H']
    for p in his:
        for q in his:
            if abs(p-q)<=2:continue
            d=min(np.linalg.norm(atoms[p][a]-atoms[q][b]) for a in ['ND1','NE2'] for b in ['ND1','NE2'])
            if d<4.2:his_clusters.add(p);his_clusters.add(q)
    for p in atoms:
        for q in atoms:
            if q<=p or abs(q-p)<=1:continue
            a=[x for n,x in atoms[p].items() if n not in BACKBONE]
            b=[x for n,x in atoms[q].items() if n not in BACKBONE]
            if a and b:
                d=float(np.min(np.linalg.norm(np.array(a)[:,None,:]-np.array(b)[None,:,:],axis=2)))
                if d<4.5:close_sc[p,q]=d
    return polar_contacts,donor_contacts,close_sc,his_clusters

def site_terms(mut,raw,summ,polars,donors,clusters):
    w,a=mut[0],mut[-1];p=int(mut[1:-1]);r=raw[p];s=summ[p]
    if w==a:return {'identity':0.}
    rsa=r.get('isolated_chain_wt_RSA')
    # Missing geometry weakens structural terms, rather than imputing zero exposure.
    geom=rsa is not None
    burial=max(0.,1.-min(1.,max(0.,rsa))/0.55) if geom else 0.
    confidence=r.get('source_asserted_AF2_pLDDT')
    reliability=max(0.25,min(1.,(confidence-50)/40)) if confidence is not None else 0.5
    phi,psi=s.get('phi_deg'),s.get('psi_deg')
    alpha=phi is not None and psi is not None and -100<phi<-35 and -80<psi<5
    beta=phi is not None and psi is not None and phi<-95 and psi>60
    gap=((B[IDX[w],IDX[w]]+B[IDX[a],IDX[a]])/2-B[IDX[w],IDX[a]])/4
    terms={'generic_similarity':0.025+0.06*gap*(0.4+0.6*burial)}
    dv=VOLUME[a]-VOLUME[w]
    terms['packing']=burial*(0.0035*max(0,dv)+0.0030*max(0,-dv))*reliability
    terms['chemistry']=0.
    if w in 'AVILMFCWY' and a in 'DEKR':
        terms['chemistry']+=2.5*burial**1.15+0.07
    elif w in 'AVILMFWY' and a in 'NQ':
        terms['chemistry']+=0.90*burial
    elif w in 'AVILMFWY' and a in 'STH':
        terms['chemistry']+=(0.35 if a in 'ST' else 0.55)*burial
    elif w=='C' and a in 'ST':terms['chemistry']+=0.15*burial
    elif w=='H' and a in 'DEKR':terms['chemistry']+=0.65*burial
    if CHARGE[w]*CHARGE[a]<0:terms['chemistry']+=0.16+0.3*burial
    if CHARGE[w] and not CHARGE[a]:
        terms['chemistry']+=0.07+0.20*burial
    if w in 'NQSTDEKRH' and a in 'AVILMFWY':terms['chemistry']+=0.18*burial
    # Existing polar contacts provide evidence for WT network disruption.
    n=len(polars[p]);retain=(a in 'NQSTDEH' if w in 'NQSTDEH' else a in 'KRH' if w in 'KR' else a in 'STY' if w in 'STY' else False)
    if n and not retain:
        terms['polar_network']=min(0.42,0.13*n)*(0.35+0.65*burial)
    elif n and w in 'DE' and a in 'NQSTH':
        terms['polar_network']=0.06*min(n,3)*burial
    # Torsion-compatible proline, glycine and other small residues.
    terms['backbone']=0.
    if a=='P':
        terms['backbone']+=0.15
        if phi is not None and (phi<-95 or phi>0):terms['backbone']+=1.15
        if donors[p]:terms['backbone']+=0.35
        if alpha:terms['backbone']+=0.40
    if w=='P':
        terms['backbone']+=0.16+0.55*burial
        if alpha:terms['backbone']+=0.12
    if a=='G':
        if phi is not None and phi>0:terms['backbone']-=0.10
        else:terms['backbone']+=0.10+0.12*burial+(0.14 if alpha else 0.08 if beta else 0.)
    if w=='G' and phi is not None and phi>0:
        terms['backbone']+=0.85*(0.35+0.65*(1-min(rsa,1))) if geom else 0.3
    if phi is not None and phi>35 and w!='G':
        if a in 'VTIP':terms['backbone']+=0.30
        elif a in 'DSN':terms['backbone']-=0.035
    # Histidine clustering is a cautious catalytic-network prior, not annotated metal binding.
    if p in clusters and w=='H' and a!='H':terms['functional_network']=0.85
    if w=='P' and any(abs(p-h)==1 for h in clusters):terms['functional_network']=0.12
    for k in list(terms):
        if k!='generic_similarity':terms[k]*=reliability
    terms['burial_descriptor']=burial
    return terms

def run():
    case=json.load(open('case.json')); packet=json.load(open('structure_evidence.json')); manifest=json.load(open('evidence_manifest.json'))
    audit=validate(case,packet,manifest); raw={x['position']:x for x in packet['raw_coordinate_evidence']['sites']};summ={x['position']:x['wt_structure'] for x in case['evidence']['sites']}
    atoms=load_atoms(); polars,donors,contacts,clusters=environment(case['reference_sequence'],atoms)
    diagnostics=[];predictions=[]
    for q in case['query']:
        muts=q['mutation'].split(':');changes={int(m[1:-1]):m for m in muts};info=[];costs=[]
        for m in muts:
            terms=site_terms(m,raw,summ,polars,donors,clusters)
            b=terms.pop('burial_descriptor',0.);cost=sum(terms.values());costs.append(cost)
            info.append({'mutation':m,'risk':cost,'burial':b,'terms':terms})
        pair_adjustment=0.
        # Bounded packing compensation only for directly contacting sidechains.
        for ii,u in enumerate(info):
            p=int(u['mutation'][1:-1]);m=u['mutation'];dv1=VOLUME[m[-1]]-VOLUME[m[0]]
            for v in info[ii+1:]:
                qpos=int(v['mutation'][1:-1]);n=v['mutation'];dv2=VOLUME[n[-1]]-VOLUME[n[0]]
                if tuple(sorted((p,qpos))) not in contacts:continue
                b=math.sqrt(u['burial']*v['burial'])
                if dv1*dv2<0:pair_adjustment-=0.12*b*min(abs(dv1),abs(dv2),40)/40
                elif dv1>0 and dv2>0:pair_adjustment+=0.12*b*min(dv1,40)/40*min(dv2,40)/40
        # Removal of proline can allow a different backbone in the same short turn.
        for p,m in changes.items():
            if m[0]=='G' and summ[p].get('phi_deg',0)>0:
                if any(abs(p-t)<=3 and n[0]=='P' and n[-1]!='P' for t,n in changes.items()):pair_adjustment-=0.12
        pair_adjustment=max(-0.25,min(0.30,pair_adjustment))
        # Strongest disruption and accumulated milder damage both matter for growth.
        total=sum(costs);worst=max(costs,default=0.)
        risk=0.80*total+0.20*worst+pair_adjustment
        score=math.tanh((1.35-risk)/2.1)
        predictions.append({'id':q['id'],'score':round(score,8)})
        diagnostics.append({'id':q['id'],'mutation':q['mutation'],'risk':risk,'sum_risk':total,'max_risk':worst,'pair_adjustment':pair_adjustment,'sites':info})
    summary='Zero-shot HIS3 growth ranking using generic substitution similarity and WT burial, packing, polar contacts, backbone compatibility, plus a weak histidine-network prior. One global scale; higher predicts better growth. All 250 mutation identities/IDs verified; reference/PDB hashes match and exact mapping covers every mutated site. No MSA or assay outcomes. Static single-chain geometry lacks cofactors/oligomer context; bounded contact adjustments are unvalidated, and rankings have substantial uncertainty.'
    result={'predictions':predictions,'summary':summary}
    assert len(predictions)==250 and len({p['id'] for p in predictions})==250
    assert all(math.isfinite(p['score']) and -1<=p['score']<=1 for p in predictions)
    Path('final_predictions.json').write_text(json.dumps(result,indent=2)+'\n')
    Path('final_diagnostics.json').write_text(json.dumps({'audit':audit,'histidine_cluster_positions':sorted(clusters),'candidates':diagnostics},indent=2)+'\n')
    print('Saved final predictions:',len(predictions),'scores; range',min(p['score'] for p in predictions),max(p['score'] for p in predictions))
    print('Histidine cluster positions:',sorted(clusters))
    for d in sorted(diagnostics,key=lambda d:d['risk'])[:15]+sorted(diagnostics,key=lambda d:d['risk'])[-15:]:
        print(d['id'],round(d['risk'],3),d['mutation'])

if __name__=='__main__':run()
