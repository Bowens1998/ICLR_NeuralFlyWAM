"""Recompute every action-source metric from saved traces before acceptance."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/sim'))
import round14_action_sources as runner

def accept(path,manifest):
    r=json.loads(Path(path).read_text());m=json.loads(Path(manifest).read_text());runner.validate_manifest(m)
    assert r['status']=='complete' and r['manifest_sha256']==runner.sha(manifest)
    assert r['row']==m['rows'][r['index']] and max(r['replay_errors'].values())<=runner.anchors.ATOL
    trace=Path(path).with_suffix('.npz');assert runner.sha(trace)==r['trace_sha256']
    specs=[x for x in m['models'] if x['dataset']==r['row']['replicate']]
    assert [x['identity'] for x in r['models']]==specs
    diag=runner.anchors.diag
    with np.load(trace,allow_pickle=False) as z:
        assert all(np.isfinite(z[k]).all() for k in z.files)
        assert z['actions'].shape==(65,50,5) and z['noise/gust'].shape==(8,50,3)
        assert np.array_equal(z['actions'][1],z['actions'][2])
        snap={'state':{'p':z['snapshot/state/p']}}
        truth={k:z['truth/'+k] for k in ['p','v','R','w','p_planner']}
        assert truth['p'].shape==(8,65,50,3)
        assert all(np.array_equal(v[:,1],v[:,2]) for v in truth.values())
        actual={k:z['logged_truth/'+k] for k in ['p','state','R']}
        at=dict(p=actual['p'][None,None],v=actual['state'][None,None,:,:3],w=actual['state'][None,None,:,9:],R=actual['R'][None,None],p_planner=(snap['state']['p']+.02*np.cumsum(actual['state'][:,:3],axis=0))[None,None])
        for i,record in enumerate(r['models']):
            state=z[f'model_{i}/state'];R=z[f'model_{i}/R'];assert state.shape==(65,50,12) and R.shape==(65,50,3,3)
            scales=diag.Normalizer.load(ROOT/record['identity']['normalizer']).metric_scales
            for label,sl,t in [('logged',slice(0,1),{k:v[:,:1] for k,v in truth.items()}),('candidate',slice(1,65),{k:v[:,1:] for k,v in truth.items()}),('actual_logged_future',slice(0,1),at)]:
                metrics=diag.horizon_metrics(state[sl],R[sl],t,snap,scales)
                for h,row in metrics.items():
                    for metric,value in row.items():
                        if isinstance(value,dict):
                            for stat,x in value.items():
                                assert np.isfinite(x) and x>=0
                                assert np.isclose(x,record['metrics'][label][h][metric][stat],rtol=1e-10,atol=1e-12)
                        else:assert np.isclose(value,record['metrics'][label][h][metric],rtol=1e-10,atol=1e-12)
    return dict(index=r['index'],result_sha256=runner.sha(path),trace_sha256=r['trace_sha256'],models=6)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--indices',type=int,nargs='+',required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    assert len(args.indices)==len(set(args.indices))
    records=[accept(args.root/f'anchor_{i:03d}.json',args.manifest) for i in args.indices]
    with args.output.open('x') as f:f.write(json.dumps(dict(accepted_anchors=len(records),records=records,manifest_sha256=runner.sha(args.manifest)),indent=2)+'\n')
    print('accepted anchors',len(records))
