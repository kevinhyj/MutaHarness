exec(open('explore2.py').read().split('\nconfigs=[]')[0])
r1=json.load(open('cv_results.json'));r2=json.load(open('cv2_results.json'))
configs=[r[-1] for r in r2[:12]]+[(r[-1],[0]*5) for r in r1[:5]]
fs=[];rg=np.random.default_rng(91379)
for rep in range(8):
 pp=rg.permutation(feedback)
 for va in np.array_split(pp,4):fs.append((np.setdiff1d(np.arange(n),va),va))
print('New partitions same observed records (not independent experimental validation):')
preds=[]
for i,cfg in enumerate(configs):
 a,b,d,e,pr=ev2(cfg,fs);preds.append(pr);print(i,'feedback',round(b,4),'double',round(d,4),'single',round(e,4))
for ids in [list(range(3)),list(range(5)),list(range(10)),list(range(12)),[0],list(range(12,17))]:
 p=np.mean([np.array([rankdata(row[32:]) for row in preds[i]]) for i in ids],0)
 print('Ensemble',ids,'feedback',np.mean([spearmanr(y[32:],row).statistic for row in p]),'double',np.mean([spearmanr(y[32:][order[32:]==2],row[order[32:]==2]).statistic for row in p]))
# A fuller challenge: train only 75% of all records.
fs2=[]
for rep in range(6):
 pp=rg.permutation(np.arange(n))
 for va in np.array_split(pp,4):fs2.append((np.setdiff1d(np.arange(n),va),va))
prs=[]
for i,cfg in enumerate(configs):
 a,b,d,e,pr=ev2(cfg,fs2);prs.append(pr)
 print('Full CV',i,'all',round(np.mean([spearmanr(y,row).statistic for row in pr]),4),'double',round(d,4),'all singles',round(np.mean([spearmanr(y[order==1],row[order==1]).statistic for row in pr]),4))
for ids in [list(range(5)),list(range(10)),list(range(12)),list(range(12,17))]:
 pr=np.mean([np.array([rankdata(row) for row in prs[i]]) for i in ids],0)
 print('Full ensemble',ids,'all',np.mean([spearmanr(y,row).statistic for row in pr]),'double',np.mean([spearmanr(y[order==2],row[order==2]).statistic for row in pr]),'single',np.mean([spearmanr(y[order==1],row[order==1]).statistic for row in pr]))
# Conservative group holdout: all substitutions to an amino acid at a hotspot withheld together.
for pos in [28,32,38,41]:
 obs={r['mutation'].split(':')[j][-1] for r in s for j,m in enumerate(r['mutation'].split(':')) if int(m[1:-1])==pos}
 oof=[];actual=[]
 for aa in sorted(obs):
  va=np.array([i for i,r in enumerate(s) if any(int(m[1:-1])==pos and m[-1]==aa for m in r['mutation'].split(':'))]);tr=np.setdiff1d(np.arange(n),va)
  pred=[]
  for cfg in configs[:10]:
   base,weights=cfg;l,ident,pw,glob,count,pair,noise,transform=base;parts0=parts[l];k=parts0[0]+ident*parts0[1]+pw*parts0[2]+glob*parts0[3]+count*parts0[4]+pair*parts0[5]+1
   for w,ex in zip(weights,extra): k+=w*ex
   co=cho_solve(cho_factor(k[np.ix_(tr,tr)]+noise*np.eye(len(tr)),lower=True),trans[transform](y[tr]));pred.append(k[np.ix_(va,tr)]@co)
  oof.extend(np.mean(pred,0));actual.extend(y[va])
 print('Unseen substitution group',pos,len(actual),spearmanr(actual,oof).statistic)
