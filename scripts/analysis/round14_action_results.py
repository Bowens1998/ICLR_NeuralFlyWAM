"""Complete-roster action-source acceptance and dataset-paired summaries."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from round14_actions_accept import accept
ROOT=Path(__file__).resolve().parents[2]

def describe(x):
    a=np.asarray(x,dtype=float);assert a.shape==(5,) and np.isfinite(a).all()
    mu=float(a.mean());half=float(t.ppf(.975,4)*a.std(ddof=1)/np.sqrt(5))
    return dict(dataset_differences=a.tolist(),mean=mu,descriptive_t95ci=[mu-half,mu+half])

def main():
    root=ROOT/'runs/results/round14_action_formal_v1';manifest=ROOT/'runs/evidence/round14_bundle/action_sources_development.json'
    accepted=[];rows=[]
    for i in range(100):
        path=root/f'anchor_{i:03d}.json';accepted.append(accept(path,manifest));r=json.loads(path.read_text())
        assert r['execution_source']=='b2234b6c0a3baf50091bae95711f7b7b3fa06eaf'
        for seed in [70,71,72]:
            arms={a:next(x for x in r['models'] if x['identity']['arm']==a and x['identity']['seed']==seed) for a in ['frozen','updated']}
            for h in ['1','5','10','25','50']:
                for metric in ['original_E','original_velocity_E','original_orientation_E','original_angular_rate_E','physical_position_l2_m','planner_position_l2_m']:
                    for stat in ['terminal','prefix_mean']:
                        scores={source:{a:arms[a]['metrics'][source][h][metric][stat] for a in arms} for source in ['logged','candidate','actual_logged_future']}
                        diff={source:v['updated']-v['frozen'] for source,v in scores.items()}
                        diff['candidate_minus_logged_interaction']=diff['candidate']-diff['logged']
                        rows.append(dict(index=i,dataset=r['row']['replicate'],wind=r['row']['mean_wind'],seed=seed,horizon=h,metric=metric,statistic=stat,scores=scores,differences=diff))
        print('accepted',i,flush=True)
    summary={}
    for wind in [6.1,8.5]:
        summary[str(wind)]={}
        for h in ['1','5','10','25','50']:
            for metric in ['original_E','original_velocity_E','original_orientation_E','original_angular_rate_E','physical_position_l2_m','planner_position_l2_m']:
                for stat in ['terminal','prefix_mean']:
                    entry={}
                    selected=[r for r in rows if r['wind']==wind and r['horizon']==h and r['metric']==metric and r['statistic']==stat]
                    for source in ['logged','candidate','actual_logged_future','candidate_minus_logged_interaction']:
                        vals=[]
                        for d in range(5):
                            group=[r for r in selected if r['dataset']==d];assert len(group)==30
                            vals.append(float(np.mean([r['differences'][source] for r in group])))
                        entry[source]=describe(vals)
                    summary[str(wind)][f'h{h}/{metric}/{stat}']=entry
    out=dict(protocol='docs/ROUND14_ACTION_SOURCE_PROTOCOL.md',accepted_anchors=100,accepted_model_records=600,acceptance=accepted,summary=summary,paired_records=rows,caveat='Conditional action-source intervention on fixed logged anchors; fresh-noise logged actions are distinct from actual realized logged future. Dataset-level descriptive intervals, not simultaneous. No full offline physical-position audit or new-architecture control result is implied.')
    with (ROOT/'reports/ROUND14_ACTION_RESULTS.json').open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
    print(json.dumps({w:s['h50/original_E/prefix_mean'] for w,s in summary.items()},indent=2))

if __name__=='__main__':main()
