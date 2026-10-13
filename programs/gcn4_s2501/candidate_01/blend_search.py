from explore import *
z2=np.load('validation_predictions2.npz');z3=np.load('validation_predictions3.npz')
tree=np.mean([z2[str(('global+domain+windows','rank',5,w,m,1.))] for w in [.3,1.] for m in ['forest','extra']],0)
joint=np.mean([z3[str((s,'rank',1,1.,'joint_rbf',g))] for s in ['smooth','small1+smooth'] for g in [.3,1]],0)
local=np.mean([z3[str(('smooth','rank',a,1.,m,g))] for a,m,g in [(100,'separate_linear',1),(1,'separate_rbf',.3)]],0)
print('tree',val(tree));print('joint',val(joint));print('local',val(local))
results=[]
for tw in [.25,.5,.75,1.]:
 for jw in [0,.25,.5,1.]:
  for lw in [0,.25,.5,1.]:
   localw=np.where(order>15,lw,jw)
   gp=joint*(1-localw)+local*localw
   p=tree*tw+gp*(1-tw)
   results.append((val(p),(tw,jw,lw)))
for v,c in sorted(results,reverse=True)[:15]:print(v,c)
