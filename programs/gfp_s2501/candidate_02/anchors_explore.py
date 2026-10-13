import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,numpy as np
from scipy.stats import spearmanr,rankdata
import proposal
from sklearn.model_selection import KFold
c=json.load(open('case.json'));s=c['support'];y=np.array([r['fitness'] for r in s]);ref=c['reference_sequence']
idx=np.array([i for i,r in enumerate(s) if r['id'].startswith('M')]);orig=np.array([i for i in range(len(s)) if i not in idx]);p=np.load('cv_predictions.npz')
names=['ET_all_10','RF_chem_2','ET_all_2','add_binary_200'];base=np.mean([np.array([rankdata(z[idx])/len(idx) for z in p[n]]) for n in names],0)
splits=[]
for rep in range(3):
 for tr,te in KFold(4,shuffle=True,random_state=150+rep).split(idx):splits.append((np.r_[orig,idx[tr]],idx[te]))
def anchor(train,test,mode):
    single={}
    for i in train:
        ms=proposal.mutations(s[i],ref)
        if len(ms)==1:
            pos,a,b=ms[0];single.setdefault(pos,[]).append((a,b,y[i]))
    out=[]
    for i in test:
        vals=[]
        for pos,a,b in proposal.mutations(s[i],ref):
            if pos not in single:continue
            if mode=='exact': ss=[z for z in single[pos] if z[1]==b]
            elif mode=='position':ss=single[pos]
            else:
                ss=single[pos]
                weights=[np.exp(-abs(proposal.HYD[b]-proposal.HYD[z[1]])/3-abs(proposal.VOL[b]-proposal.VOL[z[1]])/80-abs(proposal.charge(b)-proposal.charge(z[1])))*(.25 if (b=='P')!=(z[1]=='P') else 1) for z in ss]
                vals.append(np.average([z[2]-3.7 for z in ss],weights=weights)*max(weights));continue
            if ss:vals.append(np.mean([z[2]-3.7 for z in ss]))
        out.append(sum(vals))
    return out
for mode in ['exact','position','similarity']:
 ap=np.zeros((3,len(s)))
 for j,(tr,te) in enumerate(splits):ap[j//4,te]=anchor(tr,te,mode)
 for w in [.02,.05,.1,.2]:
  rh=[spearmanr(y[idx],base[r]+w*ap[r,idx]).statistic for r in range(3)]
  print(mode,w,np.mean(rh),np.std(rh))
