"""Post-hoc, fixed-bin A1 query boundary analysis from hash-verified saved traces."""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/sim'))
import round12_diagnostics as diag


def describe(values):
    a=np.asarray(values,dtype=float)
    assert a.shape==(5,) and np.isfinite(a).all()
    mean=float(a.mean()); half=float(t.ppf(.975,4)*a.std(ddof=1)/np.sqrt(5))
    return dict(dataset_differences=a.tolist(),mean=mean,descriptive_t95ci=[mean-half,mean+half])


def main():
    root=ROOT/'runs/results/round12a_formal_v1'
    groups=defaultdict(list);counts=defaultdict(list);sources={};records=[]
    for index in range(40):
        path=root/f'a1_{index:04d}.json';r=json.loads(path.read_text());trace=path.with_suffix('.npz')
        digest=hashlib.sha256(trace.read_bytes()).hexdigest()
        assert digest==r['trace_sha256'] and not r['pilot'] and r['status']=='complete'
        sources[path.name]=hashlib.sha256(path.read_bytes()).hexdigest();sources[trace.name]=digest
        a=r['anchors'][0];assert a['step']==100 and len(a['models'])==60
        wind=str(r['row']['wind']);prefix='anchor_100/'
        with np.load(trace) as z:
            perturb=z[prefix+'bank/perturbations']
            amplitude=np.sqrt(np.mean((perturb/np.array([.04,.08,.08]))**2,axis=(1,2)))
            masks={'zero':amplitude==0,'(0,0.5)':(amplitude>0)&(amplitude<.5),
                   '[0.5,1)':(amplitude>=.5)&(amplitude<1),'[1,infinity)':amplitude>=1}
            assert np.all(sum(m.astype(int) for m in masks.values())==1)
            snapshot={'state':{'p':z[prefix+'snapshot/state/p']}}
            truth={k:z[prefix+'evaluation/'+k][:,:64] for k in ['p','v','R','w','p_planner']}
            for label,mask in masks.items():
                counts[(wind,label)].append(int(mask.sum()))
            by_model={}
            for mi,m in enumerate(a['models']):
                if not m['model'].endswith('_r0'):continue
                d=m['dataset_replicate'];s=m['model_seed'];arm=m['model'].split('_')[-2]
                scale=diag.Normalizer.load(ROOT/f'artifacts/round11b_v1/data{d}_norm.json').metric_scales
                state=z[prefix+f'model_{mi}/state'];rotation=z[prefix+f'model_{mi}/R']
                # Verify unstratified regeneration against the accepted per-model record.
                full=diag.horizon_metrics(state,rotation,truth,snapshot,scale)
                for h in full:
                    for k in ['original_E','original_velocity_E','original_orientation_E',
                              'original_angular_rate_E','physical_position_l2_m']:
                        assert np.isclose(full[h][k]['terminal'],m['horizon_metrics'][h][k]['terminal'],rtol=1e-10,atol=1e-12)
                for label,mask in masks.items():
                    if not mask.any():continue
                    metrics=diag.horizon_metrics(state[mask],rotation[mask],{k:v[:,mask] for k,v in truth.items()},snapshot,scale)
                    by_model[(d,s,arm,label)]=metrics
            for d in range(5):
                for s in [70,71,72]:
                    for label,mask in masks.items():
                        if not mask.any():continue
                        f=by_model[d,s,'frozen',label];u=by_model[d,s,'updated',label]
                        diff={h:{k:u[h][k]['terminal']-f[h][k]['terminal'] for k in ['original_E','original_velocity_E','original_orientation_E','original_angular_rate_E','physical_position_l2_m']} for h in f}
                        records.append(dict(anchor=index,wind=wind,dataset=d,seed=s,bin=label,count=int(mask.sum()),terminal_differences=diff))
                        for h,metrics in diff.items():
                            for k,v in metrics.items():groups[(wind,label,h,k,d)].append(v)
        print('verified anchor',index,flush=True)
    summaries={}
    for wind in ['0.0','4.9','6.1','8.5']:
        summaries[wind]={}
        for label in ['zero','(0,0.5)','[0.5,1)','[1,infinity)']:
            n=counts[wind,label];entry=dict(candidate_counts_by_anchor=n,anchors_with_candidates=sum(x>0 for x in n),total_candidates=sum(n),terminal={})
            if sum(n):
                for h in ['1','5','10','25','50']:
                    entry['terminal'][h]={k:describe([np.mean(groups[wind,label,h,k,d]) for d in range(5)]) for k in ['original_E','original_velocity_E','original_orientation_E','original_angular_rate_E','physical_position_l2_m']}
            summaries[wind][label]=entry
    result=dict(post_hoc=True,protocol='docs/ROUND13_PROTOCOL.md',scope='all 40 A1 anchors; no new inference or training',amplitude='sqrt(mean((perturbation / [0.04,0.08,0.08])**2)) over 50 steps and 3 coordinates before clipping',aggregation='equal seeds and anchors within each dataset, conditional on bin being populated; all 5 datasets retained',caveat='candidate bins are observational subsets, not interventions; zero includes both identical baseline candidates at first decision; empty bins remain explicit',source_sha256=sources,wind_strata=summaries,paired_anchor_seed_records=records)
    (ROOT/'reports/ROUND13_QUERY_BOUNDARIES.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
