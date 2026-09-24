"""Summarize complete accepted position/query panels at the dataset level."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def describe(v):
 a=np.asarray(v,dtype=float);assert a.shape==(5,) and np.isfinite(a).all();mu=float(a.mean());h=float(t.ppf(.975,4)*a.std(ddof=1)/np.sqrt(5));return dict(dataset_differences=a.tolist(),mean=mu,descriptive_t95ci=[mu-h,mu+h])
def summarize(records,metrics,arms,expected):
 out={}
 for wind in [6.1,8.5]:
  out[str(wind)]={}
  for metric in metrics:
   scores={a:[] for a in arms}
   for d in range(5):
    for a in arms:
     rows=[r for r in records if r['wind']==wind and r['dataset']==d and r['arm']==a];assert len(rows)==expected
     scores[a].append(float(np.mean([r['metrics'][metric] for r in rows])))
   pairs=[('updated','frozen'),('fixed','dynamic')] if len(arms)==4 else [('fixed','dynamic')]
   out[str(wind)][metric]=dict(dataset_scores=scores,arm_means={a:float(np.mean(x)) for a,x in scores.items()},contrasts={a+'_minus_'+b:describe(np.asarray(scores[a])-scores[b]) for a,b in pairs})
 return out

def positions():
 ap=ROOT/'runs/evidence/round14_bundle/position_formal_acceptance.json';acc=json.loads(ap.read_text());assert acc['accepted_models']==60 and not acc['pilot'];assert {r['index'] for r in acc['records']}==set(range(60))
 records=[];maxdiff={}
 for row in acc['records']:
  p=ROOT/f"runs/results/round14_position_formal_v1/model_{row['index']:03d}.json";assert sha(p)==row['result_sha256'];r=json.loads(p.read_text());spec=r['identity']
  maxdiff[str(row['index'])]={k:v['max_per_window_abs_difference'] for k,v in r['checks'].items()}
  for flight in r['flights']:
   metrics={}
   for name,value in flight['metrics'].items():
    metrics[name+'/prefix_mean']=value['prefix_mean']
    for h,v in value['terminal'].items():metrics[name+'/h'+h]=v
   records.append(dict(dataset=spec['dataset'],arm=spec['arm'],seed=spec['seed'],wind=flight['wind'],flight=flight['flight'],metrics=metrics,gpu=r['gpu']))
 summary=summarize(records,list(records[0]['metrics']),['frozen','updated','fixed','dynamic'],30)
 return dict(panel='position',accepted_models=60,accepted_primary_windows=60*57000,acceptance_sha256=sha(ap),summary=summary,records=records,normalized_error_reproduction_max_abs=maxdiff,caveat='Physical position L2 meters under original Euler predicted-velocity integrator. Original models evaluated on L4, new models on RTX to reproduce accepted FP32 predictions. Paired arms within each family share hardware. Intervals descriptive across five datasets; not simultaneous.')

def queries():
 ap=ROOT/'runs/evidence/round14_bundle/query_formal_acceptance.json';acc=json.loads(ap.read_text());assert acc['accepted_anchors']==20 and not acc['pilot'];assert {r['index'] for r in acc['records']}==set(range(20))
 records=[];failures=[]
 for row in acc['records']:
  p=ROOT/f"runs/results/round14_query_formal_v1/anchor_{row['index']:03d}.json";assert sha(p)==row['sha256'];r=json.loads(p.read_text())
  for model in r['anchor']['models']:
   if model['status']!='complete':failures.append(dict(anchor=r['index'],model=model));continue
   metrics={k:model[k] for k in ['physical_solution_cost','planner_solution_cost','physical_solution_cost_excess_vs_oracle_softmin','planner_solution_cost_excess_vs_oracle_softmin','solution_threshold_exceedance_rate']}
   for h in ['1','5','10','25','50']:
    for k in ['original_E','original_velocity_E','original_orientation_E','original_angular_rate_E','physical_position_l2_m','planner_position_l2_m']:
     for stat in ['terminal','prefix_mean']:metrics[f'h{h}/{k}/{stat}']=model['horizon_metrics'][h][k][stat]
   records.append(dict(dataset=model['dataset_replicate'],arm=model['arm'],seed=model['model_seed'],wind=r['row']['wind'],anchor=r['index'],metrics=metrics))
 if failures:raise ValueError('Nonfinite models retained in accepted artifacts; do not silently average incomplete pairs. Define explicit failure analysis first.')
 summary=summarize(records,list(records[0]['metrics']),['fixed','dynamic'],30)
 return dict(panel='query',accepted_anchors=20,accepted_model_records=600,nonfinite_models=failures,acceptance_sha256=sha(ap),summary=summary,records=records,caveat='Exact original A1 histories/actions and noise, new fixed/dynamic auxiliary models; CPU FP32 inference. Actual softmin solutions evaluated under independent physical noise. Five dataset descriptive intervals, not simultaneous; not an internal mechanism identification.')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--panel',choices=['position','query'],required=True);a=ap.parse_args();result=positions() if a.panel=='position' else queries()
 with (ROOT/f'reports/ROUND14_{a.panel.upper()}_RESULTS.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 keys=['physical_position_l2_m/prefix_mean','physical_position_l2_m/h50'] if a.panel=='position' else ['physical_solution_cost','h50/original_E/prefix_mean']
 print(json.dumps({w:{k:v[k] for k in keys} for w,v in result['summary'].items()},indent=2))
