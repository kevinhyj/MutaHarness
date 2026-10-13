import json,numpy as np
from scipy.stats import spearmanr,rankdata
from model_dev import *
c=json.load(open('case.json'));s=c['support'];y=np.array([r['fitness'] for r in s]);x,z,ms=featurize(s,c['reference_sequence'])
v=json.load(open('validation_ge.json'));splits=[(np.array(a),np.array(b)) for a,b in v['splits']];preds=v['predictions']
choices=[('kernel',.3,'raw',.3),('kernel',3,'raw',.3),('kernel',3,'soft',.3),('rbf',.3,'raw',1),('rbf',.3,'raw',10),('ge',.3,.3,4.),('et',5,'raw')]
pp=np.array([[rankdata(p)/len(p) for p in preds[str(cf)]] for cf in choices])
# pp folds equal length only 63 multi? There are 63 multi; test sizes unequal.
