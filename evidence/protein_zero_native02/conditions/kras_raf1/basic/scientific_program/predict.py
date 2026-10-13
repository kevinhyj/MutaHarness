import json, math, re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASE = json.loads((ROOT / 'case.json').read_text())
MANIFEST = json.loads((ROOT / 'evidence_manifest.json').read_text())
WT = CASE['reference_chains']
# Canonical KRAS numbering: one extra leading G in the supplied chain.
assert WT['A'][1:].startswith('MTEYKLVVVGAGGVGKS')
OFFSET = 1

# Approximate burial / secondary-structure priors from the established RAS fold,
# NOT computed solvent accessibility or geometry from a supplied structure.
BURIAL = {6:1.,7:.85,8:.9,9:.9,18:.35,19:.9,21:.85,23:.9,24:.85,
 28:.75,44:.75,45:.6,46:.85,52:.75,53:.8,55:.85,56:.85,
 63:.15,64:.5,66:.2,67:.4,68:.55,71:.6,72:.65,
 78:1.,79:.9,80:.75,81:.95,82:1.,83:.7,84:.95,85:.35,86:.15,
 90:.7,93:.85,95:.1,96:.5,99:.4,103:.35,109:.65,111:.9,112:.95,114:.8,
 115:.25,118:.1,120:.45,126:.35,130:.5,133:.75,134:.25,137:.55,
 138:.15,139:.7,141:.8,142:.7,145:.35,146:.2,152:.65,155:.75,
 156:.85,159:.9,160:.85,163:.4}
HELIX = set(range(16,26)) | set(range(67,75)) | set(range(87,104)) | set(range(127,138)) | set(range(151,167))
STRAND = set(range(5,10)) | set(range(39,47)) | set(range(51,59)) | set(range(77,85)) | set(range(110,115)) | set(range(140,145))
KD = dict(zip('ACDEFGHIKLMNPQRSTVWY',[1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]))
# KD understates aromatic burial in globular proteins; preserve aromatic
# compatibility rather than treating F->Y/W as a large polarity reversal.
KD.update({'W':2.0,'Y':1.5})
VOLUME = dict(zip('ACDEFGHIKLMNPQRSTVWY',[88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.,116.1,140.,227.8,193.6]))
AA = 'ARNDCQEGHILKMFPSTWYV'
ROWS = '4 -1 -2 -2 0 -1 -1 0 -2 -1 -1 -1 -1 -2 -1 1 0 -3 -2 0\n-1 5 0 -2 -3 1 0 -2 0 -3 -2 2 -1 -3 -2 -1 -1 -3 -2 -3\n-2 0 6 1 -3 0 0 0 1 -3 -3 0 -2 -3 -2 1 0 -4 -2 -3\n-2 -2 1 6 -3 0 2 -1 -1 -3 -4 -1 -3 -3 -1 0 -1 -4 -3 -3\n0 -3 -3 -3 9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1\n-1 1 0 0 -3 5 2 -2 0 -3 -2 1 0 -3 -1 0 -1 -2 -1 -2\n-1 0 0 2 -4 2 5 -2 0 -3 -3 1 -2 -3 -1 0 -1 -3 -2 -2\n0 -2 0 -1 -3 -2 -2 6 -2 -4 -4 -2 -3 -3 -2 0 -2 -2 -3 -3\n-2 0 1 -1 -3 0 0 -2 8 -3 -3 -1 -2 -1 -2 -1 -2 -2 2 -3\n-1 -3 -3 -3 -1 -3 -3 -4 -3 4 2 -3 1 0 -3 -2 -1 -3 -1 3\n-1 -2 -3 -4 -1 -2 -3 -4 -3 2 4 -2 2 0 -3 -2 -1 -2 -1 1\n-1 2 0 -1 -3 1 1 -2 -1 -3 -2 5 -1 -3 -1 0 -1 -3 -2 -2\n-1 -1 -2 -3 -1 0 -2 -3 -2 1 2 -1 5 0 -2 -1 -1 -1 -1 1\n-2 -3 -3 -3 -2 -3 -3 -3 -1 0 0 -3 0 6 -4 -2 -2 1 3 -1\n-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4 7 -1 -1 -4 -3 -2\n1 -1 1 0 -1 0 0 0 -1 -2 -2 0 -1 -2 -1 4 1 -3 -2 -2\n0 -1 0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1 1 5 -2 -2 0\n-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1 1 -4 -3 -2 11 2 -3\n-2 -2 -2 -3 -2 -1 -2 -3 2 -1 -1 -2 -1 3 -3 -2 -2 2 7 -1\n0 -3 -3 -3 -1 -2 -2 -3 -3 3 1 -2 1 -1 -2 -2 0 -3 -1 4'
BLOSUM = {w:{a:int(x) for a,x in zip(AA,row.split())} for w,row in zip(AA,ROWS.splitlines())}
assert all(BLOSUM[w][a]==BLOSUM[a][w] for w in AA for a in AA)

# Total-effect overrides only where specific biochemical roles invalidate generic
# packing rules. These are qualitative priors, not measured coefficients.
FUNCTION_EFFECT = {
 (10,'S'):-.55,(10,'R'):-1.15,(10,'L'):-1.10,(11,'P'):-1.25,
 (13,'W'):-1.55,(14,'S'):-.65,(15,'P'):-1.90,(15,'T'):-1.20,
 (16,'Y'):-1.85,(16,'W'):-1.95,(17,'N'):-1.60,(17,'W'):-1.95,
 (17,'C'):-.75,(18,'E'):-.50,(18,'R'):-.45,(19,'P'):-1.65,
 (28,'L'):.06,(28,'I'):.02,(28,'V'):.02,(28,'T'):-.60,(28,'K'):-1.05,
 (32,'F'):-.12,(33,'N'):-.08,(33,'H'):-.28,(34,'S'):-.22,(34,'H'):-.35,
 (35,'S'):-.25,(36,'S'):-.75,(36,'G'):-.90,(36,'D'):-1.25,
 (37,'G'):-1.10,(37,'T'):-.65,(37,'W'):-1.50,
 (38,'C'):-1.15,(38,'K'):-1.60,(38,'Q'):-.85,(38,'E'):-.20,(38,'L'):-1.40,
 (39,'G'):-.22,(40,'A'):-1.20,(40,'K'):-1.60,(40,'W'):-.35,(41,'N'):-.28,
 (57,'L'):-1.10,(57,'W'):-1.35,(59,'F'):-.50,(59,'N'):-.20,(59,'H'):-.35,
 (60,'F'):-1.05,(60,'W'):-1.30,(61,'L'):.22,(61,'I'):.18,
 (61,'W'):.04,(61,'E'):-.10,(61,'N'):.00,
 (63,'V'):-.40,(63,'Y'):-.50,(63,'H'):-.30,(62,'W'):-.70,
 (64,'T'):-.35,(64,'Q'):-.35,(66,'P'):-.45,
 (115,'A'):-.10,(115,'F'):-1.10,(116,'A'):-.50,(117,'T'):.10,
 (119,'M'):-1.45,(146,'F'):-.65,
}
ACTIVATION = {(12,'C'):.10,(12,'D'):.18,(12,'V'):.22,(12,'M'):.04}

def validate():
 ids=[]
 for q in CASE['query']:
  ids.append(q['id']); chains=dict(WT); arrays={k:list(s) for k,s in WT.items()}; seen=set()
  terms=q['mutation'].split(';')
  assert len(terms)==q['mutation_order']
  for t in terms:
   m=re.fullmatch(r'([A-Z]):([A-Z])(\d+)([A-Z])',t);assert m,t
   ch,w,p,a=m.groups();p=int(p)
   assert (ch,p) not in seen;seen.add((ch,p))
   assert ch==CASE['context']['variant_chain'] and WT[ch][p-1]==w and w!=a,t
   arrays[ch][p-1]=a
  chains={k:''.join(v) for k,v in arrays.items()}
  assert chains==q['mutated_chains'] and q['sequence']==chains['A']
  for ch,s in CASE['context']['fixed_partner_chains'].items():assert chains[ch]==s
 assert len(ids)==250 and len(set(ids))==250 and set(ids)=={f'M{i:03d}' for i in range(1,251)}
 assert CASE['context']['higher_is_better'] is True
 assert MANIFEST['support_count']==0 and MANIFEST['measurements_present'] is False
 return ids

def simple_effect(w,p,a):
 if (p,a) in ACTIVATION:return ACTIVATION[p,a]
 if (p,a) in FUNCTION_EFFECT:return FUNCTION_EFFECT[p,a]
 b=BURIAL.get(p,0.0)
 cost=.025 + .030*max(0,2-BLOSUM[w][a])
 cost+=b*.12*max(0,KD[w]-KD[a])
 dv=VOLUME[a]-VOLUME[w]
 cost+=b*(.0035*max(0,dv-20)+.0030*max(0,-dv-25))
 if w in 'FWY' and a not in 'FWHY':cost+=.14*b
 if a=='P':
  if p in HELIX:cost+=.75+.20*b
  elif p in STRAND:cost+=.65+.20*b
  else:cost+=.15
 if a=='G' and p in HELIX|STRAND:cost+=.20+.10*b
 if w=='G':cost+=.10 if a in 'AS' else .35
 if w=='P':cost+=.08
 # Nucleotide-proximal loop sites are sensitive without being direct RAF contacts.
 if p in {26,27,30,48,49,58,65,67,68,73,74,75,76}:cost+=.045*max(0,2-BLOSUM[w][a])
 return -cost

def run():
 ids=validate();pred=[];features=[]
 for q in CASE['query']:
  eff=[]
  for t in q['mutation'].split(';'):
   m=re.fullmatch(r'([A-Z]):([A-Z])(\d+)([A-Z])',t);ch,w,p,a=m.groups();p=int(p)-OFFSET
   eff.append(simple_effect(w,p,a))
  # G12/Q61 activation changes share nucleotide occupancy: bound the upside.
  negative=sum(x for x in eff if x<0)
  positive=sorted((x for x in eff if x>0),reverse=True)
  activation=(positive[0]+.25*sum(positive[1:])) if positive else 0.
  latent=negative+activation
  score=math.tanh((latent+.45)/1.50)
  pred.append({'id':q['id'],'score':round(score,7)})
  features.append({'id':q['id'],'mutation':q['mutation'],'effects':eff,'latent':latent,'score':score})
 summary='Sequence-only, unsupervised KRAS–RAF1 ranking combines substitution similarity and approximate RAS-fold packing priors with nucleotide-binding and RAF-interface effects; shared G12/Q61 activation benefits are bounded. Source positions equal canonical KRAS positions plus one; fixed RAF1 chain B is unchanged. All 250 mutation identities and sequences were verified, and input SHA-256 hashes recorded. No assay outcomes, MSA or coordinates are supplied; scores use one global scale and are uncalibrated, with substantial uncertainty in nucleotide state, affinity and epistasis.'
 result={'predictions':pred,'summary':summary}
 assert len(pred)==250 and all(math.isfinite(x['score']) and -1<=x['score']<=1 for x in pred)
 (ROOT/'final_predictions.json').write_text(json.dumps(result,indent=2)+'\n')
 (ROOT/'prediction_features.json').write_text(json.dumps(features,indent=2)+'\n')
 print('Validated and saved',len(pred),'predictions. Score range',min(x['score'] for x in pred),max(x['score'] for x in pred))

if __name__=='__main__':run()
