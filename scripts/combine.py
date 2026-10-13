"""Exact-ID, blind all-three rank aggregation; never opens experimental labels."""
import argparse,json,math
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

def main():
 p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--candidate',type=Path,action='append',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if len(a.candidate)!=3:raise ValueError('Exactly three candidate files required')
 if a.out.exists():raise ValueError('Output exists; preserve it')
 case=json.loads(a.case.read_text());ids=[r['id'] for r in case['query']]
 if len(ids)!=len(set(ids)) or set(ids)&{r['id'] for r in case['support']}:raise ValueError('Unmeasured unique IDs required')
 votes=[]
 for source in a.candidate:
  scores=json.loads(source.read_text())['scores']
  if set(scores)!=set(ids):raise ValueError('Candidate must cover exactly the current unmeasured IDs')
  if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in scores.values()):raise ValueError('Finite numeric scores required')
  votes.append((rankdata([scores[i] for i in ids],method='average')-.5)/len(ids))
 values=2*np.mean(votes,axis=0)-1;out={'scores':{i:float(values[j]) for j,i in enumerate(ids)},'policy':'all-three uniform average-tie percentile ranks','candidate_files':[str(p) for p in a.candidate],'model_or_oracle_calls':0}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
