"""Optional sequence-derived feature families for development and inference."""
import numpy as np
from proposal import mutations,desc,AA,HYD,VOL,charge,features

def enhanced(records,ref,vocab=None):
    basic,vocab=features(records,ref,vocab)
    rows=[]
    # Hypothesized strand ranges; approximate annotations, no coordinate claims.
    ranges=[(11,22),(25,37),(40,50),(79,89),(94,104),(110,124),(129,142),(145,153),(161,173),(176,189),(198,208),(215,230)]
    for r in records:
        ms=mutations(r,ref);site=np.zeros((len(ref),6));wt=np.zeros((20,10));subs=np.zeros((20,20));agg=[]
        for p,a,b in ms:
            dh=(HYD[b]-HYD[a])/5;dv=(VOL[b]-VOL[a])/100;dc=charge(b)-charge(a)
            site[p-1]=[dh,abs(dh),dv,abs(dv),dc,float(b=='P')]
            dd=[abs(dh),abs(dv),abs(dc),b=='P',a=='P',b=='G',a=='G',a in 'FWY' and b not in 'FWY',max(0,-dh),float(b in 'DEKR')]
            if a in AA:wt[AA.index(a)]+=dd
            burial=.5
            beta=False
            for l,h in ranges:
                if l<=p<=h:
                    beta=True
                    same=np.mean([HYD[ref[j-1]] for j in range(l,h+1) if (j-p)%2==0])
                    opp=np.mean([HYD[ref[j-1]] for j in range(l,h+1) if (j-p)%2!=0])
                    burial=1/(1+np.exp(-(same-opp)))
                    break
            agg.append([burial,burial*abs(dh),burial*max(0,-dh),burial*abs(dv),burial*abs(dc),burial*(a in 'FWY' and b not in 'FWY'),beta*(b=='P'),float(b=='P' and not beta),float(58<=p<=75)*(b=='P'),float(a in 'FWY'),float(b in 'DEKR' and HYD[a]>1)*burial])
        aa=np.array(agg) if agg else np.zeros((1,11))
        rows.append(np.concatenate([aa.sum(0),aa.max(0),site.ravel()*.6,wt.ravel()*.5]))
    return np.concatenate([basic,np.array(rows)],1),vocab
