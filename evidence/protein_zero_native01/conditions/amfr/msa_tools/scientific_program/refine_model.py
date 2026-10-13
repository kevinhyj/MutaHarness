"""Fixed, unsupervised sequence/MSA stability prior; no outcome calibration.
Raw msa_tools scores are retained without modifying supplied frequencies.
The separate biochemical-class component aggregates exact supplied masses.
CUE helix topology is a sequence/domain prior, not coordinate evidence.
"""
import json, math
import numpy as np
from predict_variants import ROOT, load, validate_inputs, initial_predictions

# Pace-Scholtz intrinsic helix penalties (kcal/mol relative to alanine).
HELIX=dict(zip('ACDEFGHIKLMNPQRSTVWY',[0,.68,.69,.40,.54,1,.61,.41,.26,.21,.24,.65,3.16,.39,.21,.50,.66,.61,.49,.53]))
# Fauchere-Pliska side-chain partition hydrophobicities.
HYDRO=dict(zip('ACDEFGHIKLMNPQRSTVWY',[.31,1.54,-.77,-.64,1.79,0,.13,1.8,-.99,1.7,1.23,-.6,.72,-.22,-1.01,-.04,.26,1.22,2.25,.96]))
VOLUME=dict(zip('ACDEFGHIKLMNPQRSTVWY',[88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89,116.1,140,227.8,193.6]))
GROUPS=['ACFILMVWY','DE','KR','HNQ','ST','G','P']
GROUP={a:g for g in GROUPS for a in g}
POLAR=set('DEHKNQRST')
HYDROPHOBIC=set('ACFILMVWY')
CHARGED=set('DEKR')

def depth_reliability(m):
    d=m['non_gap_depth']
    if d<=0:return 0.0
    return d/(d+30)*math.sqrt(max(0,1-m['gap_fraction']))

def environments(case):
    sites={s['position']:s['msa'] for s in case['evidence']['sites']}
    # Coarse CUE three-helix topology, supported by MFP proline cap, loop
    # TRS and terminal glycine. Boundaries deliberately have weaker priors.
    helix_probability={}
    burial={}
    fits=[]
    for lo,hi in [(6,17),(22,32),(35,44)]:
        pos=np.arange(lo,hi+1)
        theta=np.deg2rad(100*(pos-lo))
        design=np.stack([np.ones(len(pos)),np.cos(theta),np.sin(theta)],axis=1)
        y=np.array([sum(sites[int(p)]['frequencies'][a] for a in HYDROPHOBIC) for p in pos])
        weight=np.sqrt([depth_reliability(sites[int(p)]) for p in pos])
        beta=np.linalg.lstsq(design*weight[:,None],y*weight,rcond=None)[0]
        amplitude=float(np.linalg.norm(beta[1:]))
        # Exposure is only a weak periodic chemistry proxy, never verified RSA.
        projection=(design[:,1:]@beta[1:])/max(.25,amplitude)
        b=np.clip(.45+.35*projection,.10,.80)
        for j,p in enumerate(pos):
            helix_probability[int(p)]=.60 if j in [0,len(pos)-1] else (.85 if j in [1,len(pos)-2] else 1.0)
            burial[int(p)]=float(b[j])
        fits.append({'range':[lo,hi],'hydrophobic_periodic_coefficients':beta.tolist(),'amplitude':amplitude})
    return sites,helix_probability,burial,fits

def final_predictions(case,evidence):
    sites,hprob,burial,fits=environments(case)
    rows=[]; diagnostics=[]
    for q,e in zip(case['query'],evidence['candidates']):
        total=0.;terms=[]
        for t in e['substitution_terms']:
            p,w,a=t['position'],t['WT'],t['ALT'];m=sites[p];f=m['frequencies']
            rel=depth_reliability(m);lr=t['sum_log_frequency_ratio']
            evo=0.;group_lr=None
            if lr is not None:
                group_lr=math.log(sum(f[x] for x in GROUP[a])/sum(f[x] for x in GROUP[w]))
                # Cap rare-residue tails: frequency ratios are not calibrated ddG.
                evo=.30*rel*(.65*np.clip(lr,-6,3)+.35*np.clip(group_lr,-6,3))
            h=hprob.get(p,0.);b=burial.get(p,.15)*h
            backbone=.75*h*(HELIX[w]-HELIX[a])
            partition=.80*b*(HYDRO[a]-HYDRO[w])
            pack=0.;hb=0.;charge=0.;cap=0.
            if rel>0 and h>0:
                mean_vol=sum(f[x]*VOLUME[x] for x in f)
                # Broad natural volume preference, modestly penalizing extremes.
                pack=-.50*b*((VOLUME[a]-mean_vol)**2-(VOLUME[w]-mean_vol)**2)/2500
                polar_mass=sum(f[x] for x in POLAR)
                if w in POLAR:
                    loss=0 if a in POLAR else (.4 if a in 'WY' else (.7 if a=='C' else 1.))
                    hb=-.60*b*polar_mass*loss
                if a in CHARGED and w not in CHARGED:charge=-.60*b
            if p==18 and w=='P':cap=-.65
            if p==45 and w=='G':cap=-.75
            if p==34 and w=='S' and a not in 'DNST':cap=-.25
            missing=0.
            if lr is None:
                # Weak generic chemistry prior for unsupported tails; L6 retains
                # its helix propensity prior, but no evolutionary/packing evidence.
                missing=-.06-.001*abs(VOLUME[a]-VOLUME[w])-.025*abs(HYDRO[a]-HYDRO[w])
            burden=-.08
            v=float(evo+backbone+partition+pack+hb+charge+cap+missing+burden)
            total+=v
            terms.append({'position':p,'WT':w,'ALT':a,'raw_msa_log_ratio':lr,'class_log_ratio':group_lr,'evidence_reliability':rel,'helix_prior_probability':h,'burial_chemistry_proxy':b,'evolutionary_component':float(evo),'backbone':backbone,'partition':partition,'packing':pack,'polar_loss':hb,'charge':charge,'cap':cap,'absent_evidence_prior':missing,'mutation_burden':burden,'latent_term':v})
        rows.append({'id':q['id'],'score':round(math.tanh(total/3),6)})
        diagnostics.append({'id':q['id'],'mutation':q['mutation'],'latent_score':total,'msa_complete':e['scores']['sum_log_frequency_ratio']['complete'],'terms':terms})
    summary=('Unsupervised stability ranking: exact supplied MSA log ratios blended with biochemical-class compatibility and modest CUE sequence-based helix, hydrophobicity, size and cap priors; high-gap evidence is attenuated. Source/WT hashes and mutation mapping verified; 239 candidates have complete MSA evidence, 11 none (positions 1–7/47). No structure, partners, assay labels or validated epistasis. Additive effects and one global tanh scale; higher is better. Relative priorities are uncertain, especially doubles and uncovered termini; scores are not calibrated assay values.')
    return {'predictions':rows,'summary':summary},{'helix_proxy_fits':fits,'candidates':diagnostics}

def main():
    case,manifest,evidence=load('case.json'),load('evidence_manifest.json'),load('msa_evidence.json')
    validate_inputs(case,manifest,evidence)
    if not (ROOT/'checkpoint_predictions.json').exists():
        (ROOT/'checkpoint_predictions.json').write_text(json.dumps(initial_predictions(case,evidence),indent=2)+'\n')
    out,diag=final_predictions(case,evidence)
    assert len(out['predictions'])==250
    assert {r['id'] for r in out['predictions']}=={q['id'] for q in case['query']}
    assert all(math.isfinite(r['score']) and -1<=r['score']<=1 for r in out['predictions'])
    (ROOT/'final_predictions.json').write_text(json.dumps(out,indent=2)+'\n')
    (ROOT/'prediction_diagnostics.json').write_text(json.dumps(diag,indent=2)+'\n')
    print('Saved and validated final_predictions.json:',len(out['predictions']))
    lookup={q['id']:q['mutation'] for q in case['query']}
    ranked=sorted(out['predictions'],key=lambda x:-x['score'])
    print('Top:',[(r['id'],lookup[r['id']],r['score']) for r in ranked[:12]])
    print('Bottom:',[(r['id'],lookup[r['id']],r['score']) for r in ranked[-8:]])
    print('Helix proxies:',diag['helix_proxy_fits'])

if __name__=='__main__':main()
