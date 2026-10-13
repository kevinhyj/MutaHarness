"""Support-only, deterministic stability ranking. No data or model file reads.

Physicochemical constants are generic amino-acid properties. Helix annotations
are sequence-derived hypotheses, not structural observations. All regression
coefficients are re-estimated from the supplied supports on every invocation.
"""
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata

AA = 'ACDEFGHIKLMNPQRSTVWY'
# Kyte-Doolittle hydropathy, approximate side-chain volume, and helix free-energy
# penalties (Pace/Scholtz; proline assigned a qualitative large penalty).
HYDRO = dict(zip(AA, [1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]))
VOL = dict(zip(AA, [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89,116.1,140,227.8,193.6]))
HELIX = dict(zip(AA, [0,.68,.69,.40,.54,1,.61,.41,.26,.21,.24,.65,3,.39,.21,.50,.66,.61,.49,.53]))

def _properties():
    p=[]
    for a in AA:
        p.append([HYDRO[a]/3, (VOL[a]-140)/50,
                  float(a in 'KR')-float(a in 'DE'), HELIX[a],
                  float(a=='P'),float(a=='G'),float(a in 'FWY'),
                  float(a in 'ST'),float(a in 'NQ'),float(a in 'IVT'),
                  float(a in 'KRH'),float(a in 'DE'),float(a=='C')])
    return np.asarray(p)
PROP=_properties()

# Deterministic inference uses no random sampling. This seed also documents the
# reproducibility convention used during local validation and exploration.
RANDOM_SEED=20261013
# Kernel amplitudes, not learned regression coefficients. Rounded configurations
# selected with repeated support-only four-fold validation, including separate
# single/double diagnostics and validation conditional on the original singles.
# Each member has an independently refitted intercept and dual coefficients.
CONFIGURATIONS=(
    {'global':31.,'context':2.,'site':15.,'identity':.04,
     'pairprop':.74,'group':.7,'environment':1.},
    {'global':3.1,'context':.76,'site':1.,'identity':.007,
     'smooth1.5':.26,'epistasis1.5':.94,'environment':.3},
    {'global':19.,'context':1.5,'site':6.5,'identity':.88,
     'pair':.16,'group':1.4,'environment':1.},
    {'global':1.4,'context':1.1,'site':1.,'identity':.57,
     'pair':.021,'group':1.4,'environment':.1},
)

def _sequence(row, ref):
    if row.get('sequence'): return row['sequence']
    s=list(ref)
    for m in row.get('mutation','').split(':'):
        if m: s[int(m[1:-1])-1]=m[-1]
    return ''.join(s)

def features(rows, ref):
    """Fixed support-independent design; site identities derived from reference."""
    n,L=len(rows),len(ref); d=PROP.shape[1]
    pairs=set()
    for row in rows:
        seq=_sequence(row,ref)
        pos=[i for i,(a,b) in enumerate(zip(ref,seq)) if a!=b]
        pairs.update((i,j) for i in pos for j in pos if i<j)
    pairs={p:i for i,p in enumerate(sorted(pairs))}
    out={k:[] for k in ['global','context','site','identity','pair','pairprop','group','aa','environment']}
    env=[]
    for i,a in enumerate(ref):
        neighbors=[j for j in [i-4,i-3,i+3,i+4] if 0<=j<L]
        h=sum(HYDRO[ref[j]] for j in neighbors)/(3*max(1,len(neighbors)))
        ch=sum(PROP[AA.index(ref[j]),2] for j in neighbors)/max(1,len(neighbors))
        env.append([np.exp(-i/4),np.exp(-(L-1-i)/4),np.exp(-abs(i-19)/3),h,ch,
                    HYDRO[a]/3,(VOL[a]-140)/50, np.sin(i*2*np.pi/3.6),np.cos(i*2*np.pi/3.6)])
    for row in rows:
        seq=_sequence(row,ref)
        muts=[(i,AA.index(a),AA.index(b)) for i,(a,b) in enumerate(zip(ref,seq)) if a!=b]
        z=np.zeros((L,d)); ident=np.zeros((L,20)); si=np.zeros(L)
        glob=np.zeros(2*d); ctx=np.zeros((5,2*d)); aa=np.zeros((4,20)); environmental=np.zeros((9,2*d))
        for i,a,b in muts:
            delta=PROP[b]-PROP[a]; z[i]=delta; ident[i,b]=1; si[i]=1
            v=np.r_[delta,np.abs(delta)]; glob+=v
            # Exposed/packed and terminal/turn contexts are soft hypotheses.
            tags=[float(i<4 or i>=37),float(18<=i<=21),float(ref[i] in 'AILMVFWY'),
                  float(ref[i] in 'KR'),float(ref[i] in 'DE')]
            ctx+=np.outer(tags,v)
            environmental+=np.outer(env[i],v)
            aa[0,b]+=1
            aa[1,b]+=float(ref[i] in 'AILMVFWY')
            aa[2,b]+=float(ref[i] in 'KR')
            aa[3,b]+=float(ref[i] in 'DE')
        # Generic pair effects and site-pair-specific smooth interactions.
        pair=np.zeros((d,d)); pairprop=np.zeros((len(pairs),d,d)); group=np.zeros(len(pairs))
        for ii in range(len(muts)):
            for jj in range(ii+1,len(muts)):
                i,a,b=muts[ii]; j,c,e=muts[jj]
                outer=np.outer(z[i],z[j]); pair+=outer
                pairprop[pairs[i,j]]=outer; group[pairs[i,j]]=1
        out['global'].append(np.r_[len(muts),glob])
        out['context'].append(ctx.ravel())
        out['site'].append(np.r_[si,z.ravel()])
        out['identity'].append(ident.ravel())
        out['pair'].append(pair.ravel())
        out['pairprop'].append(pairprop.ravel())
        out['group'].append(group.ravel())
        out['aa'].append(aa.ravel())
        out['environment'].append(environmental.ravel())
    # Drop dimensions identically zero in a support-independent reference design
    # later through dot products; the fixed flattened shape also serves queries.
    out={k:np.asarray(v,dtype=float) for k,v in out.items()}
    # Smooth, nonlinear amino-acid kernels, factored into finite feature maps.
    distances=((PROP[:,None,:]-PROP[None,:,:])**2).sum(2)
    identities=out['identity'].reshape(n,L,20)
    for length in [.7,1.5,3.]:
        kernel=np.exp(-distances/(2*length**2))
        val,vec=np.linalg.eigh(kernel)
        embed=vec*np.sqrt(np.maximum(val,0))[None,:]
        local=identities@embed
        out['smooth'+str(length)]=local.reshape(n,-1)
        interaction=np.zeros((n,len(pairs),400))
        for (i,j),index in pairs.items():
            interaction[:,index]=np.einsum('ni,nj->nij',local[:,i],local[:,j]).reshape(n,400)
        out['epistasis'+str(length)]=interaction.reshape(n,-1)
    return out

def _predict(blocks, y, ntrain, weights, ridge=1.0, target='raw'):
    K=np.zeros((len(y),len(y))); C=np.zeros((len(blocks['global'])-ntrain,ntrain))
    for key,w in weights.items():
        if w:
            X=blocks[key]; K+=w*(X[:ntrain]@X[:ntrain].T)
            C+=w*(X[ntrain:]@X[:ntrain].T)
    if target=='rank': y=(rankdata(y)-.5)/len(y)
    # An unpenalized intercept, represented by kernel centering.
    mean=float(np.mean(y)); km=K.mean(axis=0); allmean=float(K.mean())
    K=K-km[None,:]-km[:,None]+allmean
    C=C-C.mean(axis=1)[:,None]-km[None,:]+allmean
    alpha=solve(K+ridge*np.eye(ntrain), y-mean, assume_a='pos')
    return C@alpha+mean

def propose(case, assignment):
    support=[r for r in case.get('support',[]) if r.get('fitness') is not None and np.isfinite(float(r['fitness']))]
    query=case.get('query',[])
    if not query: return {'scores':{},'selected_ids':[]}
    if not support: return {'scores':{r['id']:0.0 for r in query},'selected_ids':[]}
    y=np.asarray([float(r['fitness']) for r in support])
    blocks=features(support+query,case['reference_sequence'])
    members=[]
    for weights in CONFIGURATIONS:
        p=_predict(blocks,y,len(support),weights,1.0,target='rank')
        p=np.nan_to_num(p)
        # Equal ordinal voting makes the blend insensitive to extrapolated
        # magnitudes; empirical ranks are not calibrated probabilities.
        members.append((rankdata(p,method='average')-.5)/len(p))
    p=np.mean(members,axis=0)
    score=2*(rankdata(p,method='average')-.5)/len(p)-1
    return {'scores':{r['id']:float(s) for r,s in zip(query,score)},'selected_ids':[]}
