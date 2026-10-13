"""Support-only, deterministic ordinal predictor for substitution variants.

All parameters are fitted afresh from the supplied support. No data files are
read. Amino-acid properties below are scientific descriptors, not predictions.
"""
import re
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata, norm

AA = 'ACDEFGHIKLMNPQRSTVWY'
HYDRO = dict(zip(AA, [1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]))
VOLUME = dict(zip(AA,[88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.,116.1,140.,227.8,193.6]))

def mutations(record, reference):
    if record.get('sequence') and len(record['sequence']) == len(reference):
        return [(a,i+1,b) for i,(a,b) in enumerate(zip(reference,record['sequence'])) if a != b]
    return [(a,int(p),b) for a,p,b in re.findall(r'([A-Z])(\d+)([A-Z])',record.get('mutation',''))]

def properties(a):
    return np.array([HYDRO[a]/4.5,(VOLUME[a]-140)/60,
                     float(a in 'KR')-float(a in 'DE'),float(a in 'FWY'),
                     float(a=='G'),float(a=='P'),float(a in 'STNQ')])

def features(records, reference, vocab=None):
    mm=[mutations(r,reference) for r in records]
    if vocab is None:
        vocab=sorted({(a,p,b) for m in mm for a,p,b in m})
    lookup={m:i for i,m in enumerate(vocab)}
    L=len(reference); exact=np.zeros((len(records),len(vocab)))
    site=np.zeros((len(records),L)); chem=np.zeros((len(records),56))
    for i,ms in enumerate(mm):
        chem[i,0]=len(ms)
        for a,p,b in ms:
            if (a,p,b) in lookup: exact[i,lookup[a,p,b]]=1
            site[i,p-1]=1
            x=properties(b)-properties(a)
            chem[i,1:8]+=x;chem[i,8:15]+=abs(x)
            chem[i,15+AA.index(a)]+=1;chem[i,35+AA.index(b)]+=1
    return exact,site,chem,vocab

# Relative kernel variances, distance scales and regularization strengths chosen
# by repeated support-only validation. These are configuration constants, NOT
# fitted mutation effects. Every effect coefficient is solved anew below.
CONFIGS = [{'exact': 0.7599226063222048,
  'noise': 0.5679619835657561,
  'bias': 4.880757243879451,
  'site': 1.075265649275427,
  'sitechem': 0.2220493862603227,
  'localchem': 0.11332354492719113,
  'regionchem': 0.019719231694977177,
  'globalchem': 0.06174528396100771,
  'chemlinear': 0.08970129201536071,
  'local': 1.3377686471457393,
  'region': 2.6406499247608544,
  'count': 1.2063553303187209,
  'poly': 0.031067208626197117,
  'length': 3.0,
  'form': 'rbf'},
 {'exact': 0.6497310939826432,
  'noise': 0.25947097619366843,
  'bias': 1.7907535841534832,
  'sitechem': 0.06240546346937101,
  'localchem': 2.515919395800571,
  'globalchem': 0.13026668352104054,
  'chemlinear': 0.6348121032395272,
  'local': 4.040981912965056,
  'region': 1.0544615825927321,
  'count': 0.9380311856791478,
  'poly': 0.10498793928878729,
  'length': 3.0,
  'form': 'exp'},
 {'exact': 0.7877955286911846,
  'noise': 0.3932354400854954,
  'bias': 0.6500160341021622,
  'sitechem': 0.09820215172272104,
  'localchem': 1.1194155535811765,
  'globalchem': 0.05965476238032384,
  'chemlinear': 0.9118911601913211,
  'local': 3.477894370951602,
  'region': 1.2334700141403663,
  'count': 1.5434687688486592,
  'poly': 0.034033380645988175,
  'length': 6.0,
  'form': 'exp'},
 {'exact': 1.5691640282780408,
  'noise': 0.2593331481002078,
  'bias': 1.4783778879814702,
  'sitechem': 0.0782045830472356,
  'localchem': 0.8235907719852924,
  'globalchem': 0.0949436290989648,
  'chemlinear': 0.41442890527532733,
  'local': 3.716019122241758,
  'region': 0.9270215193160881,
  'count': 1.2674898424740835,
  'poly': 0.06461707579609687,
  'length': 3.0,
  'form': 'exp'},
 {'exact': 0.5805509506464909,
  'noise': 0.24751029747416312,
  'bias': 1.5973310307874709,
  'sitechem': 0.04185471012657025,
  'localchem': 0.7627055569363894,
  'globalchem': 0.1835345813528359,
  'chemlinear': 0.4349571399557073,
  'local': 3.674785930764792,
  'region': 0,
  'count': 0.8565948361320226,
  'poly': 0.0509693192847946,
  'regionchem': 0,
  'length': 3.0,
  'form': 'exp'},
 {'exact': 0.6488883027012655,
  'noise': 0.12358425784747422,
  'bias': 0.28703886734359807,
  'site': 0.12485293025733207,
  'sitechem': 0.2305574072587178,
  'localchem': 0.03460730365636918,
  'regionchem': 0.03800570344527645,
  'chemlinear': 0.024155618455207962,
  'local': 1.0860163560404492,
  'region': 0.9846344468149812,
  'count': 0.10651279115100272,
  'length': 6.0,
  'form': 'exp'},
 {'exact': 0.5744688159673347,
  'noise': 0.16326967057421202,
  'bias': 0.485144055923755,
  'sitechem': 0.02581293967928061,
  'localchem': 1.0307673877631114,
  'globalchem': 0.11214915387172338,
  'chemlinear': 0.5854452395789999,
  'local': 3.8519261245151437,
  'region': 0.8339056939471827,
  'count': 2.4946820150564273,
  'poly': 0.10135470326333741,
  'environment': 0.09960906322534459,
  'length': 3.0,
  'form': 'exp'},
 {'exact': 0.5838043407646948,
  'noise': 1.025256506016055,
  'bias': 0.9533139418638878,
  'site': 1.1641836518769748,
  'sitechem': 0.08758006094343491,
  'localchem': 3.493315001973052,
  'regionchem': 0.015986477042537202,
  'region': 1.3708955657905137,
  'count': 1.3488407653518228,
  'environment': 0.04219285743747293,
  'length': 3.0,
  'form': 'exp'},
 {'exact': 0.48152110180033975,
  'noise': 0.5944325715418056,
  'bias': 1.1797823738122197,
  'site': 0.917482406356404,
  'sitechem': 0.1976513824870557,
  'localchem': 0.16031120602491644,
  'regionchem': 0.012130530529018167,
  'globalchem': 0.0480029362930066,
  'chemlinear': 0.2905101975687206,
  'local': 0.7241101506872479,
  'region': 0.7893807050374461,
  'count': 1.9434928181594102,
  'poly': 0.007389246450009396,
  'environment': 0.15183248019467302,
  'length': 3.0,
  'form': 'rbf'},
 {'exact': 1.061822555075391,
  'noise': 0.08101938480599757,
  'bias': 2.0492436530568114,
  'localchem': 1.591026222272178,
  'local': 1.681112334250314,
  'count': 1.4638655945961556,
  'region': 0,
  'regionchem': 0,
  'length': 6.0,
  'form': 'exp'}]

def region(p):
    # Assumed canonical Ras segment boundaries; only soft similarity features.
    # No coordinates, solvent-accessibility labels, or measured mechanism.
    bounds=[9,17,25,40,46,50,58,66,75,86,104,110,119,126,138,150,166,184,188]
    return sum(p>b for b in bounds)

def _kernel_parts(records, reference):
    mm=[mutations(r,reference) for r in records]
    vocab=sorted({m for ms in mm for m in ms})
    vi={m:i for i,m in enumerate(vocab)}
    B=np.zeros((len(records),len(vocab)))
    for i,ms in enumerate(mm):
        for m in ms:B[i,vi[m]]=1
    p=np.array([m[1] for m in vocab]);r=np.array([region(x) for x in p])
    wt=np.array([properties(m[0]) for m in vocab])
    mt=np.array([properties(m[2]) for m in vocab]);d=mt-wt
    dc=((d[:,None]-d[None,:])**2).sum(2)
    dm=((mt[:,None]-mt[None,:])**2).sum(2)
    dp=abs(p[:,None]-p[None,:]);samep=(dp==0);samer=(r[:,None]==r[None,:])
    C=features(records,reference)[2][:,:55];C[:,0]=0
    h=np.array([HYDRO[a]/4.5 for a in reference]);env=[]
    for a,pos,b in vocab:
        i=pos-1
        # Sequence-local hydropathy and broad secondary-structure hypotheses.
        env.append([np.mean(h[max(0,i-w):min(len(h),i+w+1)]) for w in [2,4,8,16]]+
                   [float(pos<=9 or 41<=pos<=46 or 51<=pos<=58 or 77<=pos<=86 or 111<=pos<=119),
                    float(18<=pos<=25 or 67<=pos<=75 or 87<=pos<=104 or 127<=pos<=138 or 151<=pos<=166),
                    float(pos>166),float(pos>184)])
    env=np.array(env);de=((env[:,None]-env[None,:])**2).sum(2)
    def aggregate(M):return B@M@B.T
    ks={'exact':B@B.T,'site':aggregate(samep),
        'sitechem':aggregate(samep*np.exp(-dm/3)),
        'localchem':aggregate(np.exp(-dp/10)*np.exp(-dc/4)),
        'regionchem':aggregate(samer*np.exp(-dc/4)),
        'globalchem':aggregate(np.exp(-dc/4)),
        'chemlinear':C@C.T/12,'region':aggregate(samer),
        'count':B.sum(1)[:,None]*B.sum(1)[None,:],
        'environment':aggregate(np.exp(-de)*np.exp(-dc/4))}
    # The quadratic kernel permits shrunk combination effects. It is positive
    # semidefinite and includes interactions between mutation similarity features.
    ks['poly']=(ks['exact']+.3*ks['site']+.1*ks['localchem'])**2
    return ks,B,dp,dc

def propose(case, assignment):
    """Refit support-only kernel ensemble and rank every supplied query.

    Raw assay outcomes and a monotone normal-rank transform are fitted in
    parallel; the latter reduces dependence on the numerical assay scale.
    No WT measurement or calibrated probability is implied. No random sampling
    is used in inference; development searches used fixed seeds (20261013,718).
    """
    support=case.get('support',[]);query=case.get('query',[])
    zero={'scores':{q['id']:0.0 for q in query},'selected_ids':[]}
    if not query or not support:return zero
    y=np.array([s['fitness'] for s in support],float);n=len(support)
    if not np.all(np.isfinite(y)):raise ValueError('Support fitness must be finite')
    if np.ptp(y)<1e-12:return zero
    records=support+query
    if not any(mutations(r,case['reference_sequence']) for r in records):return zero
    ks,B,dp,dc=_kernel_parts(records,case['reference_sequence'])
    if B.shape[1]==0:return zero
    yn=norm.ppf((rankdata(y)-.5)/n)
    raw=[];normal=[];local_cache={}
    for j,cfg in enumerate(CONFIGS):
        key=(cfg['length'],cfg['form'])
        if key not in local_cache:
            length=cfg['length']
            base=np.exp(-dp/length) if cfg['form']=='exp' else np.exp(-.5*(dp/length)**2)
            local_cache[key]=(B@base@B.T,B@(np.exp(-dp/length)*np.exp(-dc/4))@B.T)
        local,lc=local_cache[key]
        K=np.full((len(records),len(records)),cfg.get('bias',1.))
        for name,w in cfg.items():
            if name=='local':K+=w*local
            elif name=='localchem':K+=w*lc
            elif name in ks:K+=w*ks[name]
        targets=np.column_stack([y,yn]) if j<5 else y[:,None]
        coefficients=solve(K[:n,:n]+np.eye(n)*(cfg['noise']+1e-9),targets,assume_a='pos')
        pred=K[n:,:n]@coefficients
        raw.append(pred[:,0])
        if j<5:normal.append(pred[:,1])
    pred=.5*np.mean(raw,axis=0)+.5*np.mean(normal,axis=0)
    if not np.all(np.isfinite(pred)):raise ArithmeticError('Nonfinite kernel prediction')
    score=2*(rankdata(pred)-.5)/len(query)-1
    return {'scores':{q['id']:float(s) for q,s in zip(query,score)},'selected_ids':[]}
