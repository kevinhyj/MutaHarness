exec(open('confirm.py').read().split("print('New partitions")[0])
# Predeclared new-partition check with both support sampling schemes.
for title,fset,scope in [('feedback',fs,np.arange(32,n)),('full',[(np.setdiff1d(np.arange(n),va),va) for rep in range(6) for va in np.array_split(rg.permutation(np.arange(n)),4)],np.arange(n))]:
 ps=[ev2(configs[i],fset)[-1][:,scope] for i in [0,1,2,3,4,10,12,14,15]]
 ps=[np.array([rankdata(row)/len(scope) for row in p]) for p in ps]
 main=np.mean(ps[:5],0);spec=np.mean(ps[5:],0);mask=order[scope]==1
 for w in [0,.25,.5,.75,1]:
  pred=main.copy();pred[:,mask]=(1-w)*main[:,mask]+w*spec[:,mask]
  print(title,w,'all',np.mean([spearmanr(y[scope],row).statistic for row in pred]),'double',np.mean([spearmanr(y[scope][~mask],row[~mask]).statistic for row in pred]),'single',np.mean([spearmanr(y[scope][mask],row[mask]).statistic for row in pred]))
