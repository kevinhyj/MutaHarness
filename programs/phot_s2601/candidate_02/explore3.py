exec(open('explore2.py').read().split('results=[];predictions={}')[0])
results=[];predictions={};t=time.time()
# reuse independent validation splits, compare compact epistasis and global burden.
count=x.sum(1)
for family in ['46','46+60','46:59+60','46:59+60+6','46:59+60+6+91','all','burden']:
 loci=family.split('+')
 if ':' in family:
  pos,other=family.split(':');other=other.split('+')
  mask=[(v[i][1:-1]==pos and v[j][1:-1] in other) or (v[j][1:-1]==pos and v[i][1:-1] in other) for i,j in pairs]
 else:mask=[v[i][1:-1] in loci or v[j][1:-1] in loci for i,j in pairs]
 pp=xp if family=='all' else np.column_stack([count**2/16, x*(count[:,None]-8)/4]) if family=='burden' else xp[:,mask]
 for ratio in [.03,.1,.3,1]:
  k=x@x.T+ratio*pp@pp.T
  for power in [.25,.5,1]:
   for alpha in [.03,.1,.3,1]:
    evaluate(f'model_{family}_{ratio}_{power}_{alpha}',lambda tr,te,k=k,power=power,alpha=alpha:kernelpred(tr,te,k,power,alpha))
np.savez('exploration3_oof.npz',**predictions)
json.dump(results,open('exploration3_results.json','w'),indent=2)
print('Time',time.time()-t)
for r in sorted(results,key=lambda r:-(.7*r['cv']+.3*r['small']))[:25]:print(r)
