"""Sequence-only biochemical refinement. Position priors are approximate, not coordinates."""
import csv, hashlib, json, math
from pathlib import Path
import predict_gfp as base

# Confidence-weighted burial priors from established GFP topology. These are
# approximate biological knowledge, NOT observed ASA or predicted coordinates.
BURIAL={p:.30 for p in range(1,239)}
for lo,hi,inward in [(11,22,0),(25,37,1),(40,50,0),(80,89,0),(94,104,0),
                     (159,171,1),(198,207,1),(217,230,0)]:
    for p in range(lo,hi+1): BURIAL[p]=.82 if p%2==inward else .18
for p in range(1,11):BURIAL[p]=.15
for p in range(231,239):BURIAL[p]=.08
for p in range(57,76):BURIAL[p]=.8
for p in [27,29,42,44,46,48,57,61,64,68,70,71,84,88,92,94,96,98,100,106,123,125,128,130,136,137,141,143,152,161,163,165,167,178,183,201,218,220,222]:BURIAL[p]=.85
for p in [79,105,109,111,113,115,124,126,129,131,132,133,135,139,140,144,149,151,153,155,156,157,158,159,162,164,166,168,170,173,174,175,176,177,182,184,185,186,190,191,193,194,195,197,198,200,202,204,206,208,209,210,212,213,214,215,216,219,221,223,229]:BURIAL[p]=.2
for p in [146,148,150]:BURIAL[p]=.55

HYDRO=set('AVILMFWYC'); CHARGED=set('DEKR'); POLAR=set('NQSTH')
VOLUME=dict(zip('ARNDCQEGHILKMFPSTWYV',[88.6,173.4,114.1,111.1,108.5,143.8,138.4,60.1,153.2,166.7,166.7,168.6,162.9,189.9,112.7,89.0,116.1,227.8,193.6,140.0]))
# Absolute per-substitution costs override the generic chemistry model.
# Costs are heuristic severities, not measured fitness or calibrated ddG.
SPECIAL={
 'F8L':.07,'F8S':.9,'L7P':1.05,'V11A':.12,'V11E':.8,'V11M':.12,
 'I14T':1.45,'I14F':.5,'I14V':.03,'L15R':1.0,'L15Q':.5,
 'L18P':4.1,'L18Q':2.4,'V22E':4.0,'G24D':1.2,
 'F27L':2.3,'F27Y':.15,'V29E':3.9,'V29G':1.5,'V29A':.35,
 'S30P':3.3,'S30F':.65,'G31D':1.0,'G33S':.45,'G35D':.8,
 'Y39N':-.24,'Y39C':.25,'Y39H':.35,'G40S':.35,
 'L42P':4.2,'L42R':4.1,'L44P':4.1,'F46L':-.12,'F46S':3.3,
 'I47T':.28,'I47F':.3,'I47L':.12,'I47V':.04,'C48S':.25,'C48R':3.8,
 'P54L':.75,'V55A':.15,'V55G':.65,'V55E':1.8,'P56L':1.2,
 'W57R':5.0,'P58L':1.5,'P58H':1.8,'T59S':.04,'T59A':.15,'T59N':.5,
 'L60F':.8,'L60P':4.2,'V61A':.35,'V61M':.3,'V61E':3.7,
 'T62A':.4,'T62S':.05,'T63I':.85,'T63S':.06,'T63P':4.3,'L64Q':3.4,
 'S65T':-.3,'S65P':6.2,'Y66C':8.0,'Y66D':8.2,
 'V68A':.55,'V68E':4.5,'Q69R':2.2,'C70R':4.4,
 'F71L':.85,'F71Y':.12,'F71S':3.5,'S72G':-.04,'S72N':.4,'S72R':2.9,
 'R73C':.75,'Y74C':1.55,'Y74H':.6,'Y74F':.07,'P75L':.9,
 'H77R':1.15,'H77L':1.05,'M78K':1.7,'M78L':.16,'M78I':.2,'M78V':.25,
 'D82G':.65,'F83S':1.0,'F84L':.12,'F84Y':.12,'F84S':3.0,'S86P':3.3,
 'M88K':3.3,'M88T':1.5,'M88V':.1,'P89L':.65,'P89S':.6,
 'Y92F':.15,'Y92H':1.8,'Y92C':3.2,'Y92N':3.5,'Y92D':4.0,
 'V93L':.12,'V93I':.05,'V93A':.2,'Q94L':2.35,
 'R96C':6.1,'R96H':4.5,'I98T':1.6,'I98N':2.7,'I98F':.7,
 'F99S':-.22,'F99C':.08,'F99L':.13,'F100L':1.9,'F100S':4.0,'F100Y':.3,
 'D102G':.3,'D102V':.55,'D103G':.2,'D103V':.55,'D103N':.06,
 'N105S':-.08,'N105D':.08,'N105H':.4,'N105I':1.0,
 'Y106N':3.8,'Y106C':3.2,'Y106H':2.0,'V112E':2.8,
 'T118P':3.1,'T118A':.18,'T118S':.06,'N121S':.08,'N121D':.4,
 'I123F':.6,'I123V':.05,'I123T':1.4,'L125P':3.9,'L125Q':2.1,
 'I128V':.05,'I128T':1.45,'I128N':2.6,'F130L':.6,'F130Y':.12,'F130S':2.8,
 'G134R':1.05,'I136N':1.85,'I136S':1.7,'I136T':.85,'I136V':.05,
 'L137Q':2.0,'L137P':3.7,'L141Q':1.45,'L141P':3.6,
 'Y143C':1.7,'Y143N':2.0,'Y143H':.8,'Y143F':.13,
 'N144D':.1,'N144S':.07,'N144I':.4,
 'Y145F':-.2,'Y145H':.15,'Y145C':.65,'Y145S':.9,'Y145N':1.35,
 'N146S':.09,'N146D':.65,'N146K':1.4,'N146Y':1.7,
 'S147G':.25,'S147C':.15,'S147R':.7,
 'H148N':.5,'H148R':1.8,'H148L':1.3,'H148Y':1.5,'H148P':3.1,
 'N149S':.08,'N149D':.2,'N149Y':.65,'V150A':.16,'Y151F':.05,'Y151S':.7,
 'I152V':.05,'I152L':.08,'I152F':.6,'I152T':1.3,
 'M153T':-.3,'M153V':.05,'M153L':.05,'M153K':.65,'M153R':.8,
 'D155G':.35,'D155N':.08,'D155V':.6,
 'N159S':.12,'N159T':.09,'N159D':.2,'N159I':.7,'N159Y':1.05,
 'G160S':.75,'I161V':.05,'I161L':.08,'I161T':1.05,'I161F':.65,
 'V163A':-.25,'F165S':3.2,'F165L':.45,
 'I167V':.05,'I167N':1.2,'I167F':.65,
 'H169P':3.7,'H169L':.3,'H169R':1.2,
 'N170D':.3,'N170H':.25,'N170I':.55,'N170Y':.65,
 'I171V':-.23,'I171T':.13,'I171F':1.8,
 'G174A':.2,'G174S':.22,'S175G':-.15,'V176A':.07,'V176E':.65,
 'Q177R':.03,'Q177E':.12,'Q177P':.85,'L178P':3.6,'L178H':1.5,
 'Y182C':.35,'Y182H':.2,'Q183R':2.65,'Q183L':1.0,
 'I188V':.04,'I188T':.65,'I188S':.85,'I188N':1.0,'I188F':.65,
 'G189D':.8,'G191R':1.35,'P192R':.9,'V193A':.09,'V193M':.2,
 'L194P':1.45,'L194Q':.45,'L195R':1.45,'L195Q':.6,'L195V':.05,
 'P196L':.6,'P196S':.5,'N198S':.04,'N198D':.22,'N198Y':.35,
 'H199L':.5,'H199Y':.35,'H199N':.25,'H199R':.8,'Y200H':.09,'Y200C':.25,
 'L201R':3.3,'L201P':3.8,'T203A':.35,'T203I':1.25,
 'S205F':2.1,'S205P':4.0,'S205T':.14,'A206V':-.08,'A206T':.12,'L207P':3.7,
 'H217L':.15,'H217N':.12,'H217R':.2,'M218K':3.1,'M218T':1.1,'M218V':.08,'M218I':.08,
 'V219D':1.15,'V219A':.1,'L220P':4.3,'L221P':4.1,'L221Q':.85,
 'E222K':2.25,'F223L':.09,'F223Y':.09,'F223I':.13,'F223S':1.3,
 'V224E':3.2,'V224A':.35,'V224M':.4,'G228W':3.4,'G228V':1.5,
 'I229T':.75,'I229L':.1,'I229V':.08,'I229N':1.6,'H231P':.75,
 'L236P':.3,'L236Q':.2,
}

def mutation_cost(a,p,b):
    mutation=f'{a}{p}{b}'
    if mutation in SPECIAL:return SPECIAL[mutation], 'GFP_biochemical_prior'
    if p in (66,67):return 7.5,'chromophore_identity'
    if p==65:return 2.5,'chromophore_identity'
    burial=BURIAL[p]
    dissimilar=max(0,-base.BLOSUM[a,b])
    cost=.09+.045*dissimilar+burial*.11*dissimilar
    if a in HYDRO and b not in HYDRO:
        cost+=burial*(2.6 if b in CHARGED else 1.35)
    if a not in CHARGED and b in CHARGED and a not in HYDRO:
        cost+=burial*.65
    if a in CHARGED and b in CHARGED:
        if (a in 'DE')!=(b in 'DE'):cost+=burial*.45
    if b in HYDRO and a not in HYDRO:cost+=(1-burial)*.20
    if a in HYDRO and b in HYDRO:
        cost+=burial*.65*max(0,(VOLUME[a]-VOLUME[b])/100)
        cost+=burial*.50*max(0,(VOLUME[b]-VOLUME[a]-25)/100)
    if b=='P':cost+=1.1+1.9*burial
    elif b=='G' and a!='G':cost+=.15+.3*burial
    if a=='P':cost+=.4
    if a=='G':cost+=.2+.6*burial*min(1.5,(VOLUME[b]-60)/100)
    return cost,'generic_chemistry'

def main():
    case,manifest,parsed=base.load_validate()
    predictions=[]; diagnostics=[]; costs=[]
    for q in case['query']:
        individual=[mutation_cost(*m) for m in parsed[q['id']]]
        # A modest residual burden accounts for uncertainty about otherwise benign
        # changes. A weak global curvature represents finite folding margin, and
        # is not a fitted or pair-specific epistasis prediction.
        n=len(individual)
        total=sum(x[0] for x in individual)+.045*max(n-1,0)+.018*max(n-3,0)**2
        pred=math.tanh((1.8-total)/3.2)
        predictions.append({'id':q['id'],'score':round(pred,8)})
        costs.append((total,q['id'],q['mutation']))
        for m,(v,reason) in zip(parsed[q['id']],individual):
            diagnostics.append({'id':q['id'],'mutation':f'{m[0]}{m[1]}{m[2]}','cost':v,'burial_prior':BURIAL[m[1]],'basis':reason})
    summary=('Sequence-only, unsupervised GFP fluorescence ranking combines substitution chemistry, approximate barrel/core and backbone-disruption priors, chromophore/proton-network sensitivity, modest published folding-mutation priors, and mutation burden. One global monotone scale; higher is better. All 250 mutation identities and sequences were validated. No measured supports, MSA, coordinates, or partner sequences are supplied; burial and combinatorial effects remain uncertain, and scores are neither calibrated fluorescence nor validated epistasis.')
    result={'predictions':predictions,'summary':summary}
    assert len(predictions)==250 and len({p['id'] for p in predictions})==250
    assert all(math.isfinite(p['score']) and -1<=p['score']<=1 for p in predictions)
    Path('final_predictions.json').write_text(json.dumps(result,indent=2)+'\n')
    with open('mutation_diagnostics.csv','w') as out:
        writer=csv.DictWriter(out,fieldnames=list(diagnostics[0]));writer.writeheader();writer.writerows(diagnostics)
    metadata={'input_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['case.json','evidence_manifest.json']},'evidence':{'measured_support_count':0,'msa':'absent; no evolutionary score calculated','structure':'absent; topology is an uncertain knowledge prior','fixed_partners':'none supplied'},'rationale':['Refined the baseline because substitution chemistry alone treats exposed and buried changes alike and ignores GFP chromophore-specific requirements.','Applied region-dependent proline penalties and modest known folding substitutions; no assay-based fitting or pair-specific epistasis was used.'],'scale':'tanh((1.8 - total_cost)/3.2), global across all IDs','limitations':['No target outcomes to select or calibrate coefficients.','Approximate residue environment priors are not structure evidence.','Fluorescence depends on maturation, expression, photophysics and excitation, beyond folding.']}
    Path('model_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Top 20:')
    for row in sorted(costs)[:20]: print(row)
    print('Bottom 12:')
    for row in sorted(costs)[-12:]:print(row)
    print('Validated 250 finite scores; range',min(p['score'] for p in predictions),max(p['score'] for p in predictions))
if __name__=='__main__':main()
