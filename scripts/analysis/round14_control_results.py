"""Complete accepted access-control results, five dataset units and paired seeds."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[2]
def describe(x):
 a=np.asarray(x,dtype=float);assert a.shape==(5,) and np.isfinite(a).all();mu=float(a.mean());h=float(t.ppf(.975,4)*a.std(ddof=1)/np.sqrt(5));return dict(dataset_differences=a.tolist(),mean=mu,descriptive_t95ci=[mu-h,mu+h])
def main():
 ap=ROOT/'runs/evidence/round14_bundle/control_formal_acceptance.json';accept=json.loads(ap.read_text());assert accept['accepted_episodes']==600 and not accept['pilot'] and {r['index'] for r in accept['files']}==set(range(600))
 rows=[]
 for rec in accept['files']:
  p=ROOT/f"runs/results/round14_control_formal_v1/episode_{rec['index']:04d}.json";assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256'];rows.append(json.loads(p.read_text()))
 summary={}
 for wind in [6.1,8.5]:
  scores={a:[] for a in ['fixed','dynamic']};matrices={a:[] for a in scores};seedpairs=[]
  for d in range(5):
   for arm in scores:
    selected=[r for r in rows if r['row']['wind']==wind and r['row']['dataset_replicate']==d and r['row']['arm']==arm]
    assert len(selected)==30 and {(r['row']['model_seed'],r['row']['episode']) for r in selected}=={(s,e) for s in [70,71,72] for e in range(10)}
    scores[arm].append(float(np.mean([r['capped_tracking_rmse_m'] for r in selected])))
    matrices[arm].append([[next(r['capped_tracking_rmse_m'] for r in selected if r['row']['model_seed']==s and r['row']['episode']==e) for e in range(10)] for s in [70,71,72]])
   for si,s in enumerate([70,71,72]):seedpairs.append(dict(dataset=d,seed=s,fixed_minus_dynamic=float(np.mean(matrices['fixed'][d][si])-np.mean(matrices['dynamic'][d][si]))))
  delta=np.array(scores['fixed'])-scores['dynamic']
  summary[str(wind)]=dict(dataset_scores=scores,arm_means={a:float(np.mean(v)) for a,v in scores.items()},fixed_minus_dynamic=describe(delta),dataset_seed_episode_rmse=matrices,seed_pairs=seedpairs,failures={a:sum(r['failed'] for r in rows if r['row']['wind']==wind and r['row']['arm']==a) for a in scores})
 old=json.loads((ROOT/'reports/ROUND11D_RESULTS.json').read_text())
 result=dict(accepted_episodes=600,acceptance_sha256=hashlib.sha256(ap.read_bytes()).hexdigest(),summary=summary,original_historical_comparators={w:old['dataset_scores'][w] for w in ['6.1','8.5']},caveat='Fixed auxiliary minus dynamic auxiliary, parameter count matched; primary intervals use five training datasets. Original Frozen/Updated have fewer parameters, so historical cross-architecture contrasts are not pure context-access effects. No internal mechanism identified.',episodes=[dict(row=r['row'],failed=r['failed'],capped_tracking_rmse_m=r['capped_tracking_rmse_m']) for r in rows])
 with (ROOT/'reports/ROUND14_CONTROL_RESULTS.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps({w:{'means':v['arm_means'],'contrast':v['fixed_minus_dynamic'],'failures':v['failures']} for w,v in summary.items()},indent=2))
if __name__=='__main__':main()
