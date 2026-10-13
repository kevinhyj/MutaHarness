"""Deterministic support-only protein stability ranker.

All parameters are fitted anew from case['support']. No disk I/O occurs during
prediction. Residue descriptors are generic chemical constants, not predictions.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import rankdata, norm
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

AA = 'ACDEFGHIKLMNPQRSTVWY'
# Kyte-Doolittle hydropathy; residue volume; alpha helix propensity.
HYDRO = [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]
VOLUME = [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.0,116.1,140.0,227.8,193.6]
HELIX = [1.45,0.77,0.98,1.53,1.12,0.53,1.24,1.00,1.07,1.34,1.20,0.73,0.59,1.17,0.79,0.79,0.82,1.14,1.14,0.61]

def descriptors():
    a=np.array([[HYDRO[i],VOLUME[i],HELIX[i],float(x in 'KR')-float(x in 'DE'),float(x in 'FWY'),float(x=='P'),float(x=='G'),float(x in 'NQSTH'),float(x in 'IVT')] for i,x in enumerate(AA)])
    return (a-a.mean(0))/a.std(0)

def encode(records,reference):
    n=len(records); l=len(reference); d=descriptors()
    one=np.zeros((n,l,20)); prop=np.zeros((n,l,9)); site=np.zeros((n,l))
    for i,r in enumerate(records):
        seq=r.get('sequence')
        if seq is None:
            seq=list(reference)
            for m in r['mutation'].split(':'):seq[int(m[1:-1])-1]=m[-1]
        for j,(w,a) in enumerate(zip(reference,seq)):
            if w!=a and a in AA and w in AA:
                one[i,j,AA.index(a)]=1
                prop[i,j]=d[AA.index(a)]-d[AA.index(w)]
                site[i,j]=1
    return one.reshape(n,-1),prop.reshape(n,-1),site,prop.sum(1)

# BLOSUM62 is a fixed generic substitution matrix, not a fitted predictor.
BORDER='ARNDCQEGHILKMFPSTWYV'
BM=np.array([
[4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0],
[-1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3],
[-2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3],
[-2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3],
[0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
[-1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2],
[-1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2],
[0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3],
[-2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3],
[-1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3],
[-1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1],
[-1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2],
[-1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1],
[-2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1],
[-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2],
[1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2],
[0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0],
[-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3],
[-2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1],
[0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4]])


def build_kernels(records, reference):
    """Positive semidefinite hierarchical chemical/substitution kernels.

    Mutations are differences from the reference, so wild-type contributes no
    mutation feature. Residue properties are shared both within a site and
    across sites. Site-pair indicators permit conservative combination offsets.
    No experimentally established structure or contact map is assumed.
    """
    one, prop, site, glob = encode(records, reference)
    n = len(records)
    one = one.reshape(n, len(reference), 20)
    prop = prop.reshape(n, len(reference), 9)
    desc = descriptors()
    distance = ((desc[:,None]-desc[None,:])**2).mean(2)
    sim = np.exp(-distance/3.)
    order = [BORDER.index(a) for a in AA]
    blosum = BM[np.ix_(order, order)].astype(float)
    e, v = np.linalg.eigh(blosum)
    bs = (v*np.maximum(e,0.))@v.T
    bs /= np.diag(bs).mean()
    rk = np.zeros((n,n)); bk = np.zeros((n,n))
    for j, w in enumerate(reference):
        wi = AA.index(w)
        for mat, out in [(sim,rk),(bs,bk)]:
            center = mat-mat[:,wi,None]-mat[None,wi,:]+mat[wi,wi]
            out += one[:,j]@center@one[:,j].T
    sk = site@site.T
    dest = one.sum(1)
    return {'R':rk, 'B':bk, 'P':np.einsum('nla,mla->nm',prop,prop)/9.,
            'S':sk, 'G':glob@glob.T/9., 'pairs':(sk*sk-sk)/2.,
            'destination':dest@dest.T}


def ridge_prediction(kernel, y, n):
    # Gaussianized empirical ranks: ranks are computed ONLY on current supports.
    z = norm.ppf((rankdata(y)-.5)/len(y))
    mean = z.mean()
    a = cho_solve(cho_factor(kernel[:n,:n]+2.*np.eye(n)),z-mean)
    return kernel[n:,:n]@a+mean


def pair_prediction(kernel, y, n, regularization):
    # An RKHS pairwise logistic loss directly uses the signed observed ordering.
    # This is an ordinal training loss; its outputs are not assay probabilities.
    e, v = np.linalg.eigh(kernel[:n,:n])
    keep = e>1.e-8
    e=e[keep];v=v[:,keep]
    if len(e)==0 or n<2:return np.zeros(kernel.shape[0]-n)
    z=v*np.sqrt(e)
    query=kernel[n:,:n]@v/np.sqrt(e)
    i,j=np.triu_indices(n,1)
    direction=np.sign(y[i]-y[j])
    active=direction!=0
    if not np.any(active):return np.zeros(len(query))
    difference=(z[i[active]]-z[j[active]])*direction[active,None]
    def objective(beta):
        margin=difference@beta
        loss=np.logaddexp(0.,-margin).mean()+regularization*np.dot(beta,beta)/2.
        gradient=-(difference.T@expit(-margin))/len(margin)+regularization*beta
        return loss,gradient
    result=minimize(objective,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B',
                    options={'maxiter':150,'ftol':1.e-10,'gtol':1.e-7})
    return query@result.x


def propose(case, assignment):
    """Fit from this invocation's supports; return every supplied query ID.

    Model family and shrinkage constants were selected in support-only public
    validation. No IDs, outcomes, training sequences, or fitted state are stored.
    The five members combine two ordinal objectives and complementary residue
    representations. Weak site-pair offsets capture statistical combination
    effects without asserting physical contacts. No measurement is selected.
    """
    np.random.seed(0)  # Explicit reproducibility; fitting itself uses no random draws.
    supports=list(case.get('support',[]));queries=list(case.get('query',[]))
    if not queries:return {'scores':{},'selected_ids':[]}
    if not supports:return {'scores':{r['id']:0. for r in queries},'selected_ids':[]}
    n=len(supports);y=np.asarray([r['fitness'] for r in supports],dtype=float)
    # Limit BLAS threads to ensure small-matrix optimization is fast under the
    # independent executor. Algorithms are deterministic and use no random draws.
    with threadpool_limits(limits=1):
        k=build_kernels(supports+queries,case['reference_sequence'])
        common=k['S']+k['G']
        predictions=[
            ridge_prediction(k['R']+common+.1*k['pairs'],y,n),
            ridge_prediction(k['P']+common+.3*k['pairs']+.3*k['destination'],y,n),
            ridge_prediction(k['R']+.1*k['B']+common+.3*k['pairs'],y,n),
            pair_prediction(k['P']+common+.3*k['pairs'],y,n,.01),
            pair_prediction(k['R']+.5*k['B']+common+.3*k['pairs'],y,n,.003)]
    # Equal ranks within each member avoid depending on arbitrary objective scale.
    ranks=np.asarray([(rankdata(p)-.5)/len(queries) for p in predictions])
    blend=.7*ranks[:3].mean(0)+.3*ranks[3:].mean(0)
    ordinal=2.*(rankdata(blend)-.5)/len(queries)-1.
    return {'scores':{r['id']:float(s) for r,s in zip(queries,ordinal)},'selected_ids':[]}
