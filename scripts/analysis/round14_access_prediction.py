"""All-roster-gated descriptive prediction analysis for context access controls."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[2]

def describe(a):
    a=np.asarray(a,dtype=float);assert a.shape==(5,) and np.isfinite(a).all()
    mean=float(a.mean());half=float(t.ppf(.975,4)*a.std(ddof=1)/np.sqrt(5))
    return dict(dataset_differences=a.tolist(),mean=mean,descriptive_t95ci=[mean-half,mean+half])

def main():
    accepted=[]
    for d in range(5):
        a=json.loads((ROOT/f'runs/evidence/round14_bundle/round14_access_data{d}_acceptance.json').read_text())
        assert a['accepted_runs']==6 and a['accepted_npz']==12 and not a['pilot'];accepted.append(a)
    metadata={r['name']:r for r in json.loads((ROOT/'configs/sim/round11_generation.json').read_text())['rows']}
    records=[]
    for d in range(5):
        for arm in ['fixed','dynamic']:
            for seed in [70,71,72]:
                path=ROOT/f'runs/results/round14_access_data{d}_v1/r14_d{d}_{arm}_seed{seed}/eval/d2_static_ood_per_window.npz'
                assert hashlib.sha256(path.read_bytes()).hexdigest()==accepted[d]['sha256'][str(path.relative_to(ROOT))]
                with np.load(path,allow_pickle=False) as z:
                    arrays={m:z['err_'+m] for m in ['aggregate','velocity','orientation','angular_rate','displacement']}
                    ids=z['flight_ids']
                    for fid,name in enumerate(z['flight_names']):
                        row=metadata[str(name)];assert row['replicate']==d and row['group']=='static'
                        mask=ids==fid;assert mask.sum()==2850
                        metrics={m:dict(prefix_mean=float(arrays[m][mask].mean()),terminal={str(h):float(arrays[m][mask,h-1].mean()) for h in [1,5,10,25,50]}) for m in ['aggregate','velocity','orientation','angular_rate','displacement']}
                        records.append(dict(dataset=d,arm=arm,seed=seed,wind=row['mean_wind'],flight=str(name),metrics=metrics))
    summary={}
    for wind in [3.7,6.1,8.5]:
        summary[str(wind)]={}
        for metric in ['aggregate','velocity','orientation','angular_rate','displacement']:
            entry={}
            for horizon in ['prefix_mean','1','5','10','25','50']:
                scores={a:[] for a in ['fixed','dynamic']}
                for d in range(5):
                    for arm in scores:
                        r=[r for r in records if r['wind']==wind and r['dataset']==d and r['arm']==arm];assert len(r)==30
                        scores[arm].append(float(np.mean([v['metrics'][metric]['prefix_mean'] if horizon=='prefix_mean' else v['metrics'][metric]['terminal'][horizon] for v in r])))
                entry[horizon]=dict(dataset_scores=scores,fixed_minus_dynamic=describe(np.array(scores['fixed'])-scores['dynamic']))
            summary[str(wind)][metric]=entry
    result=dict(protocol='docs/ROUND14_PROTOCOL.md',contrast='fixed auxiliary context minus dynamic auxiliary context; lower error is better',parameters_per_arm=51785,uncertainty_unit='five training datasets, averaged over seeds and flights within dataset',caveat='Result-informed follow-up; descriptive non-simultaneous intervals. Parameter counts match but effective function classes need not. Prediction alone does not establish control benefit; no mechanism identification.',summary=summary,records=records)
    with (ROOT/'reports/ROUND14_ACCESS_PREDICTION.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({w:x['aggregate']['prefix_mean'] for w,x in summary.items()},indent=2))

if __name__=='__main__':main()
