"""Support-only, deterministic sequence ranking for a small stability assay.

All coefficients are fitted afresh. Amino-acid constants are generic physical
descriptors, not predictor scores. The reference is never assigned a label.
"""
import numpy as np
from scipy.linalg import solve

AA = 'ACDEFGHIKLMNPQRSTVWY'
# Kyte-Doolittle hydropathy, side-chain volume, approximate charge, and
# Pace-Scholtz helix free-energy penalty (generic scales, not assay labels).
HYD = [1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]
VOL = [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89,116.1,140,227.8,193.6]
HEL = [0,.68,.69,.40,.54,1,.61,.41,.26,.21,.24,.65,3.16,.39,.21,.50,.66,.61,.49,.53]
CHG = [0,0,-1,-1,0,0,.15,0,1,0,0,0,0,0,1,0,0,0,0,0]
RAW = np.array([HYD,VOL,CHG,HEL,[int(a=='P') for a in AA],
                [int(a=='G') for a in AA],[int(a in 'FWY') for a in AA],
                [int(a in 'IVT') for a in AA],[int(a in 'NQ') for a in AA]],float).T
PROP = (RAW-RAW.mean(0))/RAW.std(0)
INDEX = {a:i for i,a in enumerate(AA)}

def mutations(record, reference):
    if record.get('sequence'):
        return [(i,a) for i,(w,a) in enumerate(zip(reference,record['sequence'])) if a!=w]
    out=[]
    for item in record.get('mutation','').split(':'):
        if item: out.append((int(item[1:-1])-1,item[-1]))
    return out

def features(records, reference):
    """Blocks distinguish shared physics, site effects, and pair epistasis."""
    n=len(records);l=len(reference);p=PROP.shape[1]
    pairs=sorted({tuple(sorted((i,k))) for r in records
                  for x,(i,a) in enumerate(mutations(r,reference))
                  for k,b in mutations(r,reference)[x+1:]})
    pair_index={pair:j for j,pair in enumerate(pairs)}
    blocks={k:np.zeros((n,d)) for k,d in [('global',p*5+5),('site',l),
             ('local',l*p),('identity',l*20),('pattern',l*l),('interaction',p*p),
             ('context',p*6),('curved',l*20),('smooth',l*20),('smooth_short',l*20),
             ('smooth_long',l*20),('pair_local',len(pairs)*p*2),
             ('pair_interaction',len(pairs)*p*p),('severity',l),('class',l*7)]}
    # RBF descriptor embedding is a positive-definite amino-acid kernel.
    dist=((PROP[:,None,:]-PROP[None,:,:])**2).mean(2)
    eig,vec=np.linalg.eigh(np.exp(-dist/1.5))
    embedding=vec*np.sqrt(np.maximum(eig,0))
    es,vs=np.linalg.eigh(np.exp(-dist/.5));emb_short=vs*np.sqrt(np.maximum(es,0))
    el,vl=np.linalg.eigh(np.exp(-dist/4));emb_long=vl*np.sqrt(np.maximum(el,0))
    for j,r in enumerate(records):
        muts=mutations(r,reference);ds=[]
        for i,a in muts:
            ai=INDEX[a];wi=INDEX[reference[i]];d=PROP[ai]-PROP[wi];ds.append(d)
            blocks['site'][j,i]=1
            blocks['severity'][j,i]=np.sqrt(np.mean(d*d))
            for h,group in enumerate(['AILMV','FWY','KRH','DE','NQ','STC','GP']):
                blocks['class'][j,i*7+h]=float(a in group)-float(reference[i] in group)
            blocks['local'][j,i*p:(i+1)*p]=d
            blocks['identity'][j,i*20+ai]=1
            blocks['smooth'][j,i*20:(i+1)*20]=embedding[ai]-embedding[wi]
            blocks['smooth_short'][j,i*20:(i+1)*20]=emb_short[ai]-emb_short[wi]
            blocks['smooth_long'][j,i*20:(i+1)*20]=emb_long[ai]-emb_long[wi]
            nonlinear=np.r_[PROP[ai,:4]**2-PROP[wi,:4]**2,
                            PROP[ai,:4]*PROP[ai,2]-PROP[wi,:4]*PROP[wi,2],
                            PROP[ai,:4]*PROP[ai,6]-PROP[wi,:4]*PROP[wi,6],
                            np.maximum(PROP[ai,:4]-PROP[wi,:4],0),
                            np.minimum(PROP[ai,:4]-PROP[wi,:4],0)]
            blocks['curved'][j,i*20:(i+1)*20]=nonlinear
            # Sequence-derived contexts: putative helices and central turn.
            helix=float(5<=i+1<=18 or 22<=i+1<=38)
            turn=float(19<=i+1<=21)
            hyd=RAW[wi,0]/4.5;charge=RAW[wi,2]
            # These contexts use reference composition and sequence position.
            flags=[helix and reference[i] in 'AILMVFWY',
                   helix and reference[i] in 'KRDE',
                   not helix and not turn,turn,
                   i+1<=18,i+1>=22]
            for h,flag in enumerate(flags):
                blocks['context'][j,h*p:(h+1)*p]+=d*float(flag)
            blocks['global'][j,:p]+=d
            blocks['global'][j,p:2*p]+=d*helix
            blocks['global'][j,2*p:3*p]+=d*turn
            blocks['global'][j,3*p:4*p]+=d*hyd
            blocks['global'][j,4*p:5*p]+=d*charge
        blocks['global'][j,-5:]=[len(muts),len(muts)>1,
                                  sum(PROP[INDEX[a],4] for i,a in muts),
                                  sum(abs(i-(l-1)/2)/l for i,a in muts),
                                  sum(RAW[INDEX[a],2]-RAW[INDEX[reference[i]],2] for i,a in muts)**2]
        for x in range(len(muts)):
            for y in range(x+1,len(muts)):
                i,a=muts[x];k,b=muts[y]
                blocks['pattern'][j,min(i,k)*l+max(i,k)]=1
                blocks['interaction'][j]+=np.outer(ds[x],ds[y]).ravel()
                pi=pair_index[tuple(sorted((i,k)))]
                left,right=(ds[x],ds[y]) if i<k else (ds[y],ds[x])
                blocks['pair_local'][j,pi*p*2:(pi+1)*p*2]=np.r_[left,right]
                blocks['pair_interaction'][j,pi*p*p:(pi+1)*p*p]=np.outer(left,right).ravel()
    return blocks

def kernel(blocks, weights):
    n=next(iter(blocks.values())).shape[0];k=np.zeros((n,n))
    for name,w in weights.items():
        if w: k+=w*(blocks[name]@blocks[name].T)
    return k

DEFAULT = {'global':.03,'site':1.,'local':.1,'identity':.6,'pattern':.5,'interaction':.01}

# Five rounded configurations selected by repeated support-only, order-stratified
# validation. Additional split seeds and feedback holdouts checked the ensemble.
# These are regularization settings, NOT learned regression coefficients.
# All five regress transformed observed ranks: assay magnitudes are not assumed
# to equal thermodynamic free energies. Pair terms are strongly regularized.
ENSEMBLE = [
    ({'global':.012,'site':3.2,'local':.08,'pattern':.16,
      'interaction':.007,'smooth':4.,'context':.2}, .32),
    ({'global':.012,'site':3.2,'local':.08,'pattern':.16,
      'interaction':.007,'smooth':4.,'context':.01}, .13),
    ({'site':3.2,'local':.08,'pattern':.16,'interaction':.007,
      'smooth':.32,'context':.01}, .13),
    ({'global':.043,'site':2.,'local':.1,'pattern':3.5,
      'interaction':.001,'smooth':2.,'pair_interaction':.004}, .09),
    ({'global':.043,'site':2.,'local':.1,'pattern':3.5,
      'interaction':.001,'smooth_short':2.,'context':.04,
      'curved':.001,'pair_interaction':.008,'pair_local':.005}, .22),
]

def predict_kernel(k,y,train,test,alpha=1.,transform='raw'):
    yt=np.array(y)[train].copy()
    if transform=='rank':
        from scipy.stats import rankdata
        yt=(rankdata(yt)-.5)/len(yt)
    elif transform=='normal':
        from scipy.stats import rankdata,norm
        yt=norm.ppf((rankdata(yt)-.375)/(len(yt)+.25))
    elif transform=='floor':
        # Inverse smooth assay floor, used only as an unconfirmed alternative.
        yt=.5*np.log(np.expm1(np.maximum(yt+3.6,.03)/.5))
    system=k[np.ix_(train,train)]+alpha*np.eye(len(train))
    inv=solve(system,np.column_stack([yt,np.ones(len(train))]),assume_a='pos')
    center=float(inv[:,0].sum()/inv[:,1].sum());a=inv[:,0]-center*inv[:,1]
    return center+k[np.ix_(test,train)]@a

def propose(case,assignment):
    """Refit only on current supports; never request additional measurements.

    No stochastic optimization is used, so the production fit needs no random
    draws. Exploration split seeds are fixed in the retained experiment files.
    Query sequences supply covariates only and never supply training outcomes.
    """
    from scipy.stats import rankdata
    support=case.get('support',[]);query=case.get('query',[])
    if not query:return {'scores':{},'selected_ids':[]}
    if not support:return {'scores':{r['id']:0. for r in query},'selected_ids':[]}
    records=support+query; b=features(records,case['reference_sequence'])
    n=len(support);train=np.arange(n);test=np.arange(n,len(records))
    y=np.array([r['fitness'] for r in support],float)
    # Precompute shared Gram blocks once; all fitted states remain invocation-local.
    needed={key for w,alpha in ENSEMBLE for key in w}
    grams={key:b[key]@b[key].T for key in needed}
    ranks=[]
    for weights,alpha in ENSEMBLE:
        k=sum(weights[key]*grams[key] for key in weights)
        pred=predict_kernel(k,y,train,test,alpha,'rank')
        ranks.append(rankdata(pred,method='average'))
    combined=np.mean(ranks,axis=0)
    # Rank aggregation prevents arbitrary scales of members affecting the vote.
    scores=2*(rankdata(combined,method='average')-.5)/len(query)-1
    return {'scores':{r['id']:float(v) for r,v in zip(query,scores)},'selected_ids':[]}
