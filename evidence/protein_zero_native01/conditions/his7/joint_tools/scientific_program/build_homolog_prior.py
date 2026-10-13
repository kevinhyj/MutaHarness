"""Natural yeast-like homolog prior, not an assay-trained or validated epistasis model."""
import json,numpy as np

def build():
    z=np.load('alignment_parsed.npz');full=z['alignment'];ref=full[0]
    unique=np.unique(full[1:],axis=0); valid=(unique<20)&(ref<20)
    identity=((unique==ref)&valid).sum(1)/np.maximum(valid.sum(1),1)
    coverage=valid.sum(1)/(ref<20).sum()
    sel=(identity>=.5)&(coverage>=.8)&(identity<.999)
    a=unique[sel];identity=identity[sel]
    neighbors=np.zeros(len(a))
    for start in range(0,len(a),48):
        b=a[start:start+48];joint=(b[:,None,:]<20)&(a[None,:,:]<20)
        depth=joint.sum(2); matches=((b[:,None,:]==a[None,:,:])&joint).sum(2)
        neighbors[start:start+48]=((matches>=.8*depth)&(depth>=.5*(ref<20).sum())).sum(1)
    weight=1/np.maximum(neighbors,1)
    # Smooth scaffold relevance, one fixed evolutionary length scale.
    relevance=np.exp((identity-.5)/.2)
    weight=weight*relevance
    # Preserve natural effective-depth scale while changing relative scaffold relevance.
    weight*=np.sum(1/np.maximum(neighbors,1))/weight.sum()
    freq=np.zeros((220,20));depth=np.zeros(220)
    for p in range(220):
        good=a[:,p]<20; depth[p]=weight[good].sum()
        counts=np.bincount(a[good,p],weights=weight[good],minlength=20)
        freq[p]=(counts+.05)/(depth[p]+1)
    np.savez_compressed('homolog_prior.npz',alignment=a,weights=weight,frequencies=freq,depth=depth,identity=identity)
    print('Yeast-like unique homologs',len(a),'effective depth',weight.sum(),'Kish depth',weight.sum()**2/np.square(weight).sum(),'identity weighted mean',np.average(identity,weights=weight))
    c=json.load(open('case.json'));aa='ACDEFGHIKLMNPQRSTVWY';muts={}
    for q in c['query']:
        for m in q['mutation'].split(':'):muts.setdefault(int(m[1:-1]),set()).add(m[-1])
    for p in [19,56,60,62,67,76,78,80,81,82,119,142,144,154,157,161,163,165,169,178,195,203]:
        print(p,c['reference_sequence'][p-1],round(freq[p-1,aa.index(c['reference_sequence'][p-1])],3),{x:round(freq[p-1,aa.index(x)],4) for x in sorted(muts[p])})
    return a,weight,freq,depth
if __name__=='__main__':build()
