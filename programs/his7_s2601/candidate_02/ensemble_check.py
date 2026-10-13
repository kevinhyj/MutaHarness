import numpy as np
from scipy.stats import rankdata,spearmanr
p2=np.load('explore2_predictions.npz');p3=np.load('explore3_predictions.npz');y=p3['y']
a=np.r_[p2["('elastic', 0.003, 0.9, 'normal')"],p3["('full', 0.003, 0.9, 'normal')"]]
b=np.r_[p2["('elastic', 0.01, 0.1, 'normal')"],p3["('full', 0.01, 0.1, 'normal')"]]
r=np.r_[p2["('rbf', 0.8, 1, False) blend"],p3["('rbf', 0.8, 1, 'blend')"]]
k=np.r_[p2["('poly', 3, 1, 0.3) normal"],p3["('poly', 3, 1, 0.3, 'normal')"]]
def scores(v):return [round(spearmanr(y,pp).statistic,4)for pp in v]
def rank(v):return np.array([rankdata(pp)/len(pp)for pp in v])
for name,p in [('sparse',a),('ridge_sparse',b),('rbf',r),('gated_kernel',k)]:print(name,np.mean(scores(p)),scores(p))
for bw in [0,.2,.4]:
 for rw in [0,.1,.2,.3,.4]:
  p=(1-bw-rw)*rank(a)+bw*rank(b)+rw*rank(r)
  sc=scores(p);print('mix',bw,rw,np.mean(sc),'fresh',np.mean(sc[4:]),sc)
