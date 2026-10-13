"""One prespecified, unsupervised refinement of the saved checkpoint.

Reasons: a finite MSA cannot discriminate unseen conservative/radical changes;
broad homolog marginals differ from closest query backgrounds; RRM growth
function depends on RNA-facing motifs as well as fold compatibility.
No labels, parameter search, mutant coordinates, or fitted pairwise epistasis.
"""
import json,math,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial.distance import pdist,squareform
from predict import load_inputs, structural_penalty, AA, HYDRO,CHARGE

BLOSUM_ORDER='ARNDCQEGHILKMFPSTWYV'
BLOSUM_TEXT='''
4 -1 -2 -2 0 -1 -1 0 -2 -1 -1 -1 -1 -2 -1 1 0 -3 -2 0
-1 5 0 -2 -3 1 0 -2 0 -3 -2 2 -1 -3 -2 -1 -1 -3 -2 -3
-2 0 6 1 -3 0 0 0 1 -3 -3 0 -2 -3 -2 1 0 -4 -2 -3
-2 -2 1 6 -3 0 2 -1 -1 -3 -4 -1 -3 -3 -1 0 -1 -4 -3 -3
0 -3 -3 -3 9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1 1 0 0 -3 5 2 -2 0 -3 -2 1 0 -3 -1 0 -1 -2 -1 -2
-1 0 0 2 -4 2 5 -2 0 -3 -3 1 -2 -3 -1 0 -1 -3 -2 -2
0 -2 0 -1 -3 -2 -2 6 -2 -4 -4 -2 -3 -3 -2 0 -2 -2 -3 -3
-2 0 1 -1 -3 0 0 -2 8 -3 -3 -1 -2 -1 -2 -1 -2 -2 2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3 4 2 -3 1 0 -3 -2 -1 -3 -1 3
-1 -2 -3 -4 -1 -2 -3 -4 -3 2 4 -2 2 0 -3 -2 -1 -2 -1 1
-1 2 0 -1 -3 1 1 -2 -1 -3 -2 5 -1 -3 -1 0 -1 -3 -2 -2
-1 -1 -2 -3 -1 0 -2 -3 -2 1 2 -1 5 0 -2 -1 -1 -1 -1 1
-2 -3 -3 -3 -2 -3 -3 -3 -1 0 0 -3 0 6 -4 -2 -2 1 3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4 7 -1 -1 -4 -3 -2
1 -1 1 0 -1 0 0 0 -1 -2 -2 0 -1 -2 -1 4 1 -3 -2 -2
0 -1 0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1 1 5 -2 -2 0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1 1 -4 -3 -2 11 2 -3
-2 -2 -2 -3 -2 -1 -2 -3 2 -1 -1 -2 -1 3 -3 -2 -2 2 7 -1
0 -3 -3 -3 -1 -2 -2 -3 -3 3 1 -2 1 -1 -2 -2 0 -3 -1 4
'''
B0=np.array([[int(x) for x in line.split()] for line in BLOSUM_TEXT.strip().splitlines()]); assert np.array_equal(B0,B0.T)
ix=[BLOSUM_ORDER.index(a) for a in AA]; B=B0[np.ix_(ix,ix)]
K=2.**(B/2); K/=K.sum(axis=1,keepdims=True)
AA_INDEX={a:i for i,a in enumerate(AA)}


def near_homolog_profile(c):
 """Deduplicate full ordered domains; weight near rows by RRM2 cluster density.

 The close subset is chosen using OTHER RRM domains, not candidate changes.
 Query-lowercase insertion columns are excluded; K164 remains absent evidence.
 Raw row count, diverse effective depth and native supplied Neff are distinct.
 """
 seqs=[]; seq=''
 for line in open('raw_evidence/PABP_YEAST_full_11-26-2021_b07.a2m'):
  if line.startswith('>'):
   if seq: seqs.append(seq)
   seq=''
  else: seq+=line.strip()
 seqs.append(seq)
 ref=c['reference_sequence']; assert seqs[0].upper()==ref
 assert all(len(s)==len(ref) for s in seqs)
 querymask=np.array([a.isupper() for a in seqs[0]])
 ordered=np.array([35<=p<=410 for p in range(1,len(ref)+1)])&querymask
 # Collapsing exact duplicates removes repeated deposited copies.
 seen=set(); unique=[]
 for seq in seqs:
  key=''.join(a if a in AA else '-' for a,b in zip(seq,ordered) if b)
  if key not in seen: unique.append(seq); seen.add(key)
 X=np.array([list(s) for s in unique]); wt=np.array(list(ref))
 valid=np.isin(X,list(AA))
 background=querymask & np.array([(35<=p<=120) or (205<=p<=410) for p in range(1,len(ref)+1)])
 coverage=valid[:,background].mean(1)
 identity=((X[:,background]==wt[background])&valid[:,background]).sum(1)/np.maximum(1,valid[:,background].sum(1))
 select=(identity>=.7)&(coverage>=.8)
 X=X[select]; valid=valid[select]
 local=querymask & np.array([126<=p<=200 for p in range(1,len(ref)+1)])
 encoded=np.array([[AA_INDEX.get(a,20) for a in x[local]] for x in X],dtype=np.uint8)
 distances=squareform(pdist(encoded,metric='hamming'))
 # All selected rows have complete local coverage in this input. Do not
 # count matching gaps as residues for an MSA with incomplete local coverage.
 assert valid[:,local].all()
 weights=1/np.maximum(1,(distances<=.2+1e-10).sum(axis=1))
 profiles={}; depths={}
 for p in range(126,201):
  if not querymask[p-1]: continue
  v=valid[:,p-1]; depth=float(weights[v].sum()); counts=np.array([weights[(X[:,p-1]==a)&v].sum() for a in AA])
  if depth>0:
   # Small explicitly new prior in this new model; supplied fixed-tool
   # frequencies and scores are preserved unchanged.
   profiles[p]=(counts+0.05)/(depth+1.)
   depths[p]=depth
 audit={'raw_row_count':len(seqs),'unique_ordered_domain_rows':len(unique),'close_unique_rows':len(X),'close_effective_depth':float(weights.sum()),'selection':'identity >=0.70 and coverage >=0.80 on RRM1/RRM3/RRM4; RRM2 not used for retrieval','weighting':'inverse number of close rows >=80% RRM2 identity','query_lowercase_sites_excluded':[p for p in range(126,201) if not querymask[p-1]]}
 return profiles,depths,audit


def functional_penalty(p,w,a):
 # Canonical RRM RNP2 and RNP1 aromatic nucleobase stacking sites,
 # established from sequence motifs. RNA is not present in the WT coordinates.
 if p in (129,168,170):
  assert w=='F'
  return 0.0 if a in 'FYW' else .6 if a=='H' else 1.4
 if p in (131,166):
  assert w=='K'
  return 0.0 if a in 'KR' else .35 if a=='H' else 1.0 if a in 'DE' else .5
 return 0.


def main():
 c,m,s,sites,changes,hashes=load_inputs()
 near,depths,msa_audit=near_homolog_profile(c)
 rawsites={r['position']:r for r in s['raw_coordinate_evidence']['sites']}
 assert all(rawsites[p]['WT']==row['WT'] for p,row in sites.items())
 assert not any(r['cross_chain_heavy_atom_contacts'] for r in rawsites.values())
 diagnostics=[]
 for q in c['query']:
  total=0.; terms=[]
  for p,w,a in changes[q['id']]:
   row=sites[p]; f=np.array([row['msa']['frequencies'][b] for b in AA]); depth=row['msa']['non_gap_depth']
   assert abs(f.sum()-1)<1e-8
   observed=depth>0 and row['msa']['homolog_support']!='none'
   if observed:
    # Small background-specific contribution; attenuate when effective close
    # support is scarce. This is model weighting, not measured assay fitting.
    near_fraction=.30*depths.get(p,0)/(depths.get(p,0)+5)
    mixed=(1-near_fraction)*f+near_fraction*near.get(p,f)
    # BLOSUM substitution kernel regularizes finite-profile zero/rare counts.
    # Different from and explicitly separate from deterministic native tools.
    smoothed=.85*mixed+.15*(mixed@K)
    evo=math.log(smoothed[AA_INDEX[a]]/smoothed[AA_INDEX[w]])
    native_llr=math.log(f[AA_INDEX[a]]/f[AA_INDEX[w]])
   else:
    near_fraction=0.; native_llr=None
    # Absent insertion-alignment evidence stays absent. A weak generic prior.
    evo=.20*(B[AA_INDEX[w],AA_INDEX[a]]-B[AA_INDEX[w],AA_INDEX[w]])
   stpen=structural_penalty(row,w,a)
   phi=row['wt_structure'].get('phi_deg')
   # Asn is also common in positive-phi turns; bulky replacements lose that
   # compatibility even when the WT is not glycine (not a mutant torsion).
   if phi is not None and phi>30 and w in 'ND' and a not in 'GND': stpen+=.65
   funpen=functional_penalty(p,w,a)
   value=evo-.85*stpen-funpen
   total+=value
   terms.append({'position':p,'WT':w,'ALT':a,'msa_observed':observed,'native_msa_log_ratio':native_llr,'near_profile_fraction':near_fraction,'regularized_evolution_or_missing_prior':evo,'WT_structure_penalty':stpen,'RNA_motif_penalty':funpen,'combined':value,'WT_RSA':rawsites[p]['isolated_chain_wt_RSA'],'WT_pLDDT':row['wt_structure']['pLDDT']})
  diagnostics.append({'id':q['id'],'mutation':q['mutation'],'raw_score':float(total),'terms':terms,'msa_coverage':sum(t['msa_observed'] for t in terms)/len(terms)})
 vals=np.array([d['raw_score'] for d in diagnostics])
 # One affine+tanh transform for the entire panel. Location/scale only display
 # priorities, without implying experimental WT or neutral fitness calibration.
 center=float(np.median(vals)); scale=float(max(1,(np.percentile(vals,90)-np.percentile(vals,10))/2))
 preds=[{'id':d['id'],'score':round(float(np.tanh((d['raw_score']-center)/scale)),7)} for d in diagnostics]
 out={'predictions':preds,'summary':'Label-free additive ranking combines regularized homolog frequencies, a modest close-homolog background prior, and confidence-aware WT burial/backbone and RRM RNA-contact penalties. All hashes, exact residue mapping, mutation identities and 250 IDs verified. Ten candidates have incomplete MSA coverage; missing terms use a weak biochemical prior. One global score scale; no measured calibration, RNA-bound structure, mutant relaxation or validated epistasis, so uncertainty is substantial.'}
 assert len(preds)==250 and len({p['id'] for p in preds})==250
 assert all(math.isfinite(p['score']) and -1<=p['score']<=1 for p in preds)
 Path('final_predictions.json').write_text(json.dumps(out,indent=2)+'\n')
 Path('prediction_diagnostics.json').write_text(json.dumps({'stage':'final','hashes':hashes,'msa_model':msa_audit,'scale':{'center':center,'scale':scale},'complete_MSA_candidates':sum(d['msa_coverage']==1 for d in diagnostics),'structure_mapping':'exact whole-chain sequence; all 577 WT residues observed','fixed_partner_context':'No RNA/other chain present; RNA motif priors do not estimate affinity to altered partners.','coefficients_prespecified':{'broad_plus_close_profile_kernel_fraction':.15,'close_profile_max_fraction':.30,'close_depth_shrinkage':5,'structure_coefficient':.85},'candidates':diagnostics},indent=2)+'\n')
 with open('model_rationale.md','a') as fh:
  fh.write('\nOne substantive refinement, without outcome testing or parameter search:\n1. Apply a 15% BLOSUM62 substitution kernel to finite homolog marginals: unseen conservative and radical substitutions otherwise receive nearly identical floor penalties. Preserve all deterministic tool values and report the new model separately.\n2. Add a small diversity-weighted close-homolog profile, selected on the other RRM domains. Raw observations show context differences at H134/P135/F168/H172/E195; this addresses broad taxonomic marginal mismatch. Query-lowercase K164 remains absent MSA evidence.\n3. Add canonical RRM RNA motif penalties and positive-phi Asn-turn compatibility to represent assay function and resolve the glycine-only backbone prior. No geometric RNA affinity, mutant relaxation, or experimentally validated epistasis claimed.\n4. Use one global median/quantile tanh display scale; monotonicity preserves all ranking and does not estimate measured neutrality.\n')
 print('Final validated:',len(preds),'IDs; missing-MSA candidates',sum(d['msa_coverage']<1 for d in diagnostics))
 print('MSA supplemental audit:',msa_audit)
 print('Global score transform:',center,scale,'range',min(p['score'] for p in preds),max(p['score'] for p in preds))
 print('Lowest:',[(d['id'],d['mutation'],round(d['raw_score'],2)) for d in sorted(diagnostics,key=lambda z:z['raw_score'])[:12]])
 print('Highest:',[(d['id'],d['mutation'],round(d['raw_score'],2)) for d in sorted(diagnostics,key=lambda z:z['raw_score'])[-12:]])
if __name__=='__main__':main()
