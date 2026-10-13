"""Additional sequence-only descriptors used in scientific exploration."""
import numpy as np
from proposal import AA,PROP,mutations,region

def align(a,b):
 n,m=len(a),len(b);dp=np.zeros((n+1,m+1));dp[:,0]=-2.5*np.arange(n+1);dp[0,:]=-2.5*np.arange(m+1);trace=np.zeros((n+1,m+1),int)
 for i in range(1,n+1):
  for j in range(1,m+1):
   sim=3. if a[i-1]==b[j-1] else .5-.3*np.sum((PROP[AA.index(a[i-1])]-PROP[AA.index(b[j-1])])**2)
   vals=[dp[i-1,j-1]+sim,dp[i-1,j]-2.5,dp[i,j-1]-2.5];k=int(np.argmax(vals));dp[i,j]=vals[k];trace[i,j]=k
 out={};i,j=n,m
 while i>0 and j>0:
  k=trace[i,j]
  if k==0:out[i-1]=j-1;i-=1;j-=1
  elif k==1:i-=1
  else:j-=1
 return out

def repeats(ref):
 out={}
 if len(ref)<400:return out
 for lo,hi,others in [(126,166,[(39,79),(220,259),(321,362)]),(167,200,[(80,116),(260,294),(363,398)])]:
  a=ref[lo-1:hi]
  for blo,bhi in others:
   b=ref[blo-1:bhi];match=align(a,b)
   for i,j in match.items():out.setdefault(lo+i,[]).append(b[j])
 return out

def descriptors(records,ref):
 rep=repeats(ref);out=[]
 for rec in records:
  vv=[]
  for p,a,b in mutations(rec,ref):
   ai,bi=AA.index(a),AA.index(b);x,z=PROP[ai],PROP[bi];d=z-x
   neighbors=rep.get(p,[a]);r=np.array([PROP[AA.index(t)] for t in neighbors])
   mismatch=(r-z)**2-(r-x)**2
   pref=np.r_[mismatch.mean(axis=0),np.mean([b!=t for t in neighbors])-np.mean([a!=t for t in neighbors]),np.mean(np.all(r==x,axis=1)),np.mean(r,axis=0)]
   v=np.r_[p/100, float(p%2), x,z,d,np.abs(d),pref]
   vv.append(v)
  out.append(vv)
 return out

def extra_kernels(records,ref):
 desc=descriptors(records,ref);n=len(desc);out=[]
 for sigma in [.5,1,2,4]:
  k=np.zeros((n,n));g=np.zeros((n,n));sm=np.zeros((n,n))
  for i in range(n):
   for j in range(i+1):
    for a in desc[i]:
     for b in desc[j]:
      sim=np.exp(-np.sum((a[9:16]-b[9:16])**2)/(2*sigma*sigma)) # target properties
      if a[0]==b[0]:k[i,j]+=sim
      spatial=np.exp(-((a[0]-b[0])*100/3)**2/2)
      sm[i,j]+=sim*spatial
      g[i,j]+=np.exp(-np.sum((a[16:]-b[16:])**2)/(2*sigma*sigma))
    k[j,i]=k[i,j];g[j,i]=g[i,j];sm[j,i]=sm[i,j]
  out.extend([k,sm,g])
 # Homologous repeat conservation / residue context sharing, without external data.
 f=np.array([np.sum([v[30:] for v in row],axis=0) if row else np.zeros(16) for row in desc])
 out.append(f@f.T)
 return np.array(out),desc

def tree_features(records,ref):
 desc=descriptors(records,ref); out=[]
 for rec,vs in zip(records,desc):
  v=np.array(vs)
  if len(v)==0:v=np.zeros((1,46))
  p=np.array([t[0]*100 for t in vs])
  # Order-independent aggregation and sorted mutation representations.
  ss=np.vstack([v, v[:1]])[:2]
  regions=np.zeros(12)
  for z in p:regions[region(z)]+=1
  out.append(np.r_[len(v),v.sum(axis=0),v.min(axis=0),v.max(axis=0),ss.flatten(),regions])
 return np.array(out)
