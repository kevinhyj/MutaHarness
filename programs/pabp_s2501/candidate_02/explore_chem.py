from explore import *
c=json.load(open('case.json'));recs=c['support'];y=np.array([r['fitness'] for r in recs]);orders=np.array([r['mutation_order'] for r in recs]);ks=kernels(recs,c['reference_sequence']);sp=splits(y,orders)
p,t,pos,ms=basic_features(recs,c['reference_sequence']);pi={v:i for i,v in enumerate(pos)}
n=len(recs);z=np.zeros((n,len(pos),30));dest=np.zeros((n,len(pos),20));deltaaa=np.zeros_like(dest)
for i,m in enumerate(ms):
 for po,a,b in m:
  u,v=AA.index(a),AA.index(b);d=props[v]-props[u]
  z[i,pi[po]]=np.r_[d,abs(d),props[v]]
  dest[i,pi[po],v]=1;deltaaa[i,pi[po],v]=1;deltaaa[i,pi[po],u]=-1
for name,zz in [('delta',z[:,:,:10]),('abs',z[:,:,10:20]),('dest',z[:,:,20:]),('aa',dest),('aadelta',deltaaa)]:
 zz=zz.reshape(n,-1);ks[name]=zz@zz.T/(10 if name not in ['aa','aadelta'] else 1)
for w in [.25,1,4,16]:
 d=((props[:,None,:]-props[None,:,:])**2).mean(-1)
 aa=np.exp(-d/(2*w))
 ks['rbfaa'+str(w)]=np.einsum('ipa,ab,jpb->ij',dest,aa,dest,optimize=True)
 ks['rbfdelta'+str(w)]=np.einsum('ipa,ab,jpb->ij',deltaaa,aa,deltaaa,optimize=True)
results=[]
for name in ['chem','delta','abs','dest','aa','aadelta']+['rbfaa'+str(w) for w in [.25,1,4,16]]+['rbfdelta'+str(w) for w in [.25,1,4,16]]:
 for scale in [1,3,10]:
  for s in [0,.3,1]:
   k=ks['p']+s*ks['s1.5']+scale*ks[name]+.1*ks['global']
   for a in [.1,.5,2]:
    for mode in ['raw','sqrt','log0.1','log0.3']:
     results.append((evaluate(k,y,sp,a,mode),name,scale,s,a,mode))
results.sort(reverse=True);print('CHEM OPTIONS',results[:25],flush=True)
json.dump(results,open('chem_results.json','w'))
# A new seed set is reserved to assess a small ensemble after selection.
np.savez('explore_kernels.npz',**ks)
