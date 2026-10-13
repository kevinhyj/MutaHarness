"""Fixed, unsupervised WT-neighborhood homolog profile, no outcome fitting."""
import json
import numpy as np
from scipy.spatial.distance import cdist
from inspect_alignment import read_alignment

AA='ACDEFGHIKLMNPQRSTVWY'

def local_profile(return_homologs=False):
    a,w,m,rows=read_alignment()
    code={aa:i for i,aa in enumerate(AA+'-')}
    x=np.array([[code.get(aa,20) for aa in row] for row in a],dtype=np.int8)
    wt=np.array([code.get(aa,20) for aa in w],dtype=np.int8)
    covered=(x<20).sum(axis=1)
    ident=(x==wt).sum(axis=1)/covered
    take=(ident>=.55)&(covered>=.8*len(w))
    x=x[take];identity=ident[take]
    # Identity-neighborhood weighting corrects uneven phylogenetic sampling.
    # Pair coverage is >=80% for all selected rows; ignore gap matches.
    nbr=np.zeros(len(x))
    for start in range(0,len(x),100):
        y=x[start:start+100]
        matches=((y[:,None,:]==x[None,:,:])&(y[:,None,:]<20)).sum(axis=2)
        paircoverage=((y[:,None,:]<20)&(x[None,:,:]<20)).sum(axis=2)
        nbr[start:start+100]=((matches/np.maximum(paircoverage,1))>=.8).sum(axis=1)
    weights=1/np.maximum(nbr,1)
    # Broad homology cutoff followed by a smooth WT-context preference.
    weights*=np.exp((identity-.55)/.12)
    weights/=weights.max()
    counts=np.zeros((len(w),20))
    for i in range(20):counts[:,i]=(weights[:,None]*(x==i)).sum(axis=0)
    depth=counts.sum(axis=1)
    broad=json.load(open('raw_evidence/msa_profile.json'))
    prior=np.array([[broad['positions'][str(int(p))]['frequency'][aa] for aa in AA] for p in m])
    # Five prior sequence equivalents guard sparsity without changing supplied scores.
    frequency=(counts+5*prior)/(depth[:,None]+5)
    diagnostics={'rows':len(x),'weighted_depth':float(weights.sum()),'kish_effective_rows':float(weights.sum()**2/(weights**2).sum()),'identity_min':float(identity.min()),'identity_max':float(identity.max())}
    if return_homologs:
        return frequency,m,diagnostics,x,weights
    return frequency,m,diagnostics

if __name__=='__main__':
    f,m,d=local_profile();print(d)
    c=json.load(open('case.json'));bypos={int(pos):f[i] for i,pos in enumerate(m)}
    for pos in [7,8,11,19,56,60,62,67,78,80,81,82,86,96,119,120,141,142,144,157,161,163,164,165,169,172,180,195,203,206]:
        r=c['reference_sequence'][pos-1];mut=set(q['sequence'][pos-1] for q in c['query'])-{r}
        print(pos,r,'WT %.3f'%bypos[pos][AA.index(r)],' '.join('%s %.3f'%(alt,bypos[pos][AA.index(alt)]) for alt in sorted(mut)))
