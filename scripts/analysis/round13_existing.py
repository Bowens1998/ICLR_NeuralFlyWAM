"""Post-review descriptive paired analyses; no training, selection or new p-values."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[2]
NAMES=['ROUND11B_RESULTS','ROUND11D_RESULTS','ROUND11C_RESULTS','ROUND12A1_RESULTS','ROUND12A2_RESULTS']
inputs={name:json.loads((ROOT/'reports'/f'{name}.json').read_text()) for name in NAMES}

def describe(x):
    x=np.asarray(x,dtype=float)
    assert x.ndim==1 and np.isfinite(x).all()
    mean=float(x.mean());se=x.std(ddof=1)/np.sqrt(len(x))
    interval=list(t.interval(.95,len(x)-1,loc=mean,scale=se)) if se else [mean,mean]
    return dict(values=x.tolist(),mean=mean,descriptive_t95ci=interval,n=len(x))

joint={}
for wind,condition in [('6.1','static_50wind'),('8.5','static_70wind')]:
    p=inputs['ROUND11B_RESULTS']['dataset_scores'][condition]
    c=inputs['ROUND11D_RESULTS']['dataset_scores'][wind]['capped_rmse_m']
    dx=np.asarray(p['updated_r0'])-p['frozen_r0'];dy=np.asarray(c['updated_r0'])-c['frozen_r0']
    assert dx.shape==dy.shape==(5,)
    joint[wind]=dict(prediction=describe(dx),control=describe(dy),joint_reversal_count=int(((dx<0)&(dy>0)).sum()),pairs=[dict(dataset=i,prediction=float(x),control=float(y)) for i,(x,y) in enumerate(zip(dx,dy))])
    seed_pairs=[]
    control_rows=inputs['ROUND11D_RESULTS']['episode_diagnostics']
    prediction_rows=inputs['ROUND11B_RESULTS']['episode_records']
    for d in range(5):
        for seed in [70,71,72]:
            ps={arm:[r['aggregate'] for r in prediction_rows if r['dataset']==d and r['seed']==seed and r['condition']==condition and r['arm']==arm+'_r0'] for arm in ['frozen','updated']}
            cs={arm:[r['capped_tracking_rmse_m'] for r in control_rows if r['row']['kind']=='model' and r['row']['dataset_replicate']==d and r['row']['model_seed']==seed and r['row']['wind']==float(wind) and r['row']['model']==f'r11b_d{d}_{arm}_r0'] for arm in ['frozen','updated']}
            assert all(len(x)==10 for x in [*ps.values(),*cs.values()])
            seed_pairs.append(dict(dataset=d,seed=seed,prediction=float(np.mean(ps['updated'])-np.mean(ps['frozen'])),control=float(np.mean(cs['updated'])-np.mean(cs['frozen']))))
    matrix=[]
    for d in range(5):
        differences=[]
        for episode in range(10):
            cs={arm:[r['capped_tracking_rmse_m'] for r in control_rows if r['row']['kind']=='model' and r['row']['dataset_replicate']==d and r['row']['wind']==float(wind) and r['row']['episode']==episode and r['row']['model']==f'r11b_d{d}_{arm}_r0'] for arm in ['frozen','updated']}
            assert all(len(x)==3 for x in cs.values())
            differences.append(float(np.mean(cs['updated'])-np.mean(cs['frozen'])))
        matrix.append(differences)
    assert np.allclose(np.mean(matrix,axis=1),dy)
    for d in range(5):
        assert np.isclose(np.mean([r['prediction'] for r in seed_pairs if r['dataset']==d]),dx[d])
    joint[wind].update(seed_pairs=seed_pairs,seed_joint_reversal_count=sum(r['prediction']<0 and r['control']>0 for r in seed_pairs),control_dataset_by_episode=matrix,control_episode_conditional=describe(np.mean(matrix,axis=0)))
scores=inputs['ROUND11C_RESULTS']['scores'];location={}
for arm in ['frozen','updated']:
    # (inputLN - neither) - (readoutLN - neither) = inputLN - readoutLN.
    location[arm]=describe(np.asarray(scores[arm+'_r1o0'])-scores[arm+'_r0o1'])
location['interaction']=describe(np.asarray(location['updated']['values'])-location['frozen']['values'])
query={}
for w,group in inputs['ROUND12A1_RESULTS']['wind_strata'].items():
    c=group['contrasts']['updated_minus_frozen_no_ln']
    query[w]={str(h):{metric:c[f'h{h}/{metric}/terminal'] for metric in ['original_E','original_velocity_E','original_orientation_E','original_angular_rate_E','physical_position_l2_m']} for h in [1,5,10,25,50]}
mc={w:g['paired_mc']['updated_minus_frozen_no_ln']['physical_solution_cost'] for w,g in inputs['ROUND12A1_RESULTS']['wind_strata'].items()}
result=dict(protocol='docs/ROUND13_PROTOCOL.md',post_hoc=True,no_new_p_values=True,joint=joint,placement_direct=location,a1_terminal_horizon=query,a1_physical_cost_mc=mc,source_sha256={n:hashlib.sha256((ROOT/'reports'/f'{n}.json').read_bytes()).hexdigest() for n in NAMES})
(ROOT/'reports/ROUND13_EXISTING_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'joint':joint,'placement_direct':location},indent=2))
