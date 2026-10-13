import json,numpy as np
from scipy.stats import spearmanr,rankdata
r=json.load(open('validation_results.json'));z=np.load('validation_predictions.npz');mid=z['mid'];y=z['Y'][mid];p=z['predictions'][:,:,mid]
pr=np.array([[rankdata(a)/len(a) for a in pp] for pp in p])
def score(a):return np.mean([spearmanr(y,aa).statistic for aa in a])
base=(pr[0]+pr[1])/2
print('base',score(base))
for typ in ['ridge','poly','rbf','extra','forest','boost']:
 choices=[i for i,v in enumerate(r) if v['config'][0]==typ]; best=max(choices,key=lambda i:score(.8*base+.2*pr[i]));print(typ,r[best]['config'],score(.8*base+.2*pr[best]))
best=[]
for i,v in enumerate(r):
 for w in [.1,.25,.5]:best.append((score((1-w)*base+w*pr[i]),w,i,v['config']))
for rr in sorted(best,reverse=True)[:20]:print(rr)
