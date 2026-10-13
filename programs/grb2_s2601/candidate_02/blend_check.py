import json,numpy as np
from scipy.stats import rankdata,spearmanr
z=np.load('explore2_results.npz');results=json.load(open('explore2_results.json'));pr=z['preds'];double=z['doubles'];y=z['y'];order=z['order']
# Best rank and logit families, averaging near-optimal configurations rather than
# fitting an unconstrained stacker to the small validation set.
for k in [1,3,5,10,20,40]:
 idx=order[:k];blend=np.mean([[rankdata(pr[i,j,double]) for j in range(2)] for i in idx],0)
 print('top',k,[spearmanr(y[double],blend[j]).statistic for j in range(2)])
for k in [1,3,5,10]:
 ri=[i for i in order if results[i][3]==1][:k];li=[i for i in order if results[i][3] in [2,3,4]][:k]
 r=np.mean([[rankdata(pr[i,j,double]) for j in range(2)] for i in ri],0);l=np.mean([[rankdata(pr[i,j,double]) for j in range(2)] for i in li],0)
 for w in [.25,.5,.75,1]:print('rank/logit',k,w,[spearmanr(y[double],w*r[j]+(1-w)*l[j]).statistic for j in range(2)])
