"""Recompute new-model MPPI weighted solutions and costs from retained traces."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/sim'))
import round14_access_queries as runner
D=runner.diag

def accept(path,manifest,source):
 m=json.loads(Path(manifest).read_text());runner.validate(m);r=json.loads(Path(path).read_text());assert r['status']=='complete' and r['manifest_sha256']==D.sha256(manifest) and r['execution_source']==source
 assert r['original_anchor']==m['anchors'][r['index']] and r['compatibility_checked_original_models']==30
 trace=Path(path).with_suffix('.npz');assert r['trace_sha256']==D.sha256(trace)
 records=r['anchor']['models'];assert len(records)==30
 assert [(x['dataset_replicate'],x['arm'],x['model_seed']) for x in records]==[(x['dataset'],x['arm'],x['seed']) for x in m['models']]
 with np.load(trace,allow_pickle=False) as z:
  assert all(np.isfinite(z[k]).all() for k in z.files)
  snap=dict(state={'p':z['snapshot/state/p']},p_ref=z['snapshot/p_ref'],v_ref=z['snapshot/v_ref'])
  truth={k:z['evaluation/'+k] for k in ['p','v','R','w','p_planner']};pert=z['evaluation/perturbations']
  complete=[x for x in records if x['status']=='complete'];assert all(x['status'] in ['complete','nonfinite_prediction'] for x in records)
  assert truth['p'].shape==(8,65+len(complete),50,3)
  costs=D.costs(truth,snap,pert)
  for label in ['physical','planner']:np.testing.assert_allclose(costs[label],z['evaluation_cost/'+label],rtol=1e-10,atol=1e-12)
  for i,rec in enumerate(records):
   assert rec['checkpoint_sha256']==m['models'][i]['checkpoint_sha256']
   if rec['status']!='complete':continue
   state=z[f'model_{i}/state'];rotation=z[f'model_{i}/R'];assert state.shape==(64,50,12)
   pred=D.planner_cost(state[...,:3],snap['state']['p'],snap['p_ref'],snap['v_ref'],z['bank/perturbations'],np.array([.04,.08,.08]));weights,solution=D.softmin_solution(pred,z['bank/perturbations'])
   np.testing.assert_allclose(pred,z[f'model_{i}/predicted_cost'],rtol=1e-10,atol=1e-12)
   np.testing.assert_allclose(weights,z[f'model_{i}/weights'],rtol=1e-10,atol=1e-12)
   np.testing.assert_allclose(solution,z[f'model_{i}/solution_perturbation'],rtol=1e-10,atol=1e-12)
   si=64+rec['solution_index'];np.testing.assert_allclose(pert[si],solution,rtol=1e-10,atol=1e-12)
   for label in ['physical','planner']:
    assert np.isclose(costs[label][:,si].mean(),rec[label+'_solution_cost'],rtol=1e-10,atol=1e-12)
    assert np.isclose((costs[label][:,si]-costs[label][:,64]).mean(),rec[label+'_solution_cost_excess_vs_oracle_softmin'],rtol=1e-10,atol=1e-12)
   scales=D.Normalizer.load(ROOT/m['models'][i]['config']['data']['normalizer_path']).metric_scales
   metrics=D.horizon_metrics(state,rotation,{k:v[:,:64] for k,v in truth.items()},snap,scales)
   for h,row in metrics.items():
    for metric,values in row.items():
     if isinstance(values,dict):
      for stat,v in values.items():assert np.isclose(v,rec['horizon_metrics'][h][metric][stat],rtol=1e-10,atol=1e-12)
  assert sorted(x['solution_index'] for x in complete)==list(range(1,len(complete)+1))
 return dict(index=r['index'],sha256=D.sha256(path),trace_sha256=r['trace_sha256'],nonfinite_models=len(records)-len(complete))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--source',required=True);ap.add_argument('--pilot',action='store_true');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 indices=[0,10] if a.pilot else range(20);records=[accept(a.root/f'anchor_{i:03d}.json',a.manifest,a.source) for i in indices]
 with a.output.open('x') as f:f.write(json.dumps(dict(accepted_anchors=len(records),records=records,source=a.source,pilot=a.pilot),indent=2)+'\n')
 print('accepted anchors',len(records))
