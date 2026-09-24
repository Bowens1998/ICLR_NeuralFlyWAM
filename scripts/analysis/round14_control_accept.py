"""Accept complete or failed access-control episodes without dropping failures."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/sim'))
import round14_access_control as p

def accept(path,manifest,source,pilot):
    m=json.loads(Path(manifest).read_text());p.validate(m);r=json.loads(Path(path).read_text());claim=json.loads(Path(path).with_suffix('.claim').read_text())
    assert len(source)==40 and r['execution_source']==source
    index=r['index'];row=m['rows'][index];assert r['row']==row and r['source_sha256']==m['source_sha256'] and r['schema']==m['schema']
    assert r['pilot']==pilot and r['manifest_sha256']==p.sha(manifest)
    assert claim=={k:r[k] for k in ['index','manifest_sha256','execution_source','pilot']}
    if pilot:assert index in m['pilot_indices']
    spec=next(x for x in m['models'] if (x['dataset'],x['arm'],x['seed'])==(row['dataset_replicate'],row['arm'],row['model_seed']))
    assert r['checkpoint_sha256']==spec['checkpoint_sha256']
    errors=r['errors_m'];assert len(errors)<=1500 and all(x is None or (np.isfinite(x) and x>=0) for x in errors)
    missing=[i for i,x in enumerate(errors) if x is None];assert not missing or (r['failed'] and missing==[len(errors)-1])
    expected=p.engine.protocol.score([np.nan if x is None else x for x in errors],r['failed'],1500)
    for k,v in expected.items():
        if v is None or isinstance(v,bool):assert r[k]==v
        else:assert np.isclose(r[k],v,rtol=1e-10,atol=1e-12)
    actions=np.asarray(r['actions'],dtype=float)
    if actions.size==0:actions=np.empty((0,5))
    assert actions.ndim==2 and actions.shape[1]==5 and np.isfinite(actions).all()
    assert np.all((actions[:,0]>=.05-1e-12)&(actions[:,0]<=1+1e-12))
    assert np.allclose(np.linalg.norm(actions[:,1:],axis=-1),1,rtol=0,atol=1e-6)
    latency=np.asarray(r['planning_seconds']);assert np.isfinite(latency).all() and (latency>0).all()
    if not r['failed']:
        assert r['failure_reason'] is None and r['failure_step'] is None and len(actions)==1600 and len(errors)==len(latency)==1500
    else:
        state_reasons={'nonfinite_state','tracking_error','position_norm','velocity_norm','body_rate_norm'}
        reason=r['failure_reason'];assert reason in state_reasons|{'nonfinite_candidate_action','nonfinite_prediction','nonfinite_action'}
        step=r['failure_step'];assert type(step)==int and 0<=step<1600
        assert len(actions)==step+int(reason in state_reasons)
        assert len(errors)==max(0,len(actions)-100)
        assert len(latency)==max(0,len(actions)-100)+int(reason=='nonfinite_action' and step>=100)
    assert r['observed_effort_steps']==len(errors)
    if r['warmup_state'] is not None:
        old=next(x for x in p.engine.protocol.design()['rows'] if x['kind']=='model' and x['dataset_replicate']==0 and x['model_seed']==70 and x['model']=='r11b_d0_frozen_r0' and x['wind']==row['wind'] and x['episode']==row['episode'])
        ref=json.loads((ROOT/f"runs/results/round11d_formal_v1/episode_{old['index']:04d}.json").read_text())
        for k,v in r['warmup_state'].items():np.testing.assert_allclose(v,ref['warmup_state'][k],rtol=0,atol=1e-10)
    return dict(index=index,sha256=p.sha(path),failed=r['failed'],elapsed_seconds=r['elapsed_seconds'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--source',required=True);ap.add_argument('--pilot',action='store_true');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    m=json.loads(Path(a.manifest).read_text());indices=m['pilot_indices'] if a.pilot else range(600)
    files=[accept(a.root/f'episode_{i:04d}.json',a.manifest,a.source,a.pilot) for i in indices]
    with a.output.open('x') as f:f.write(json.dumps(dict(accepted_episodes=len(files),pilot=a.pilot,files=files,manifest_sha256=p.sha(a.manifest),source=a.source),indent=2)+'\n')
    print('accepted',len(files),'failed',sum(x['failed'] for x in files))
