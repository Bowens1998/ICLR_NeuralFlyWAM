"""Hash-bound paired future-action intervention on original logged anchors."""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).resolve().parent))
import round14_logged_anchors as anchors
from latent_aero_wam.evaluation.evaluator import load_model_from_checkpoint

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def sources():
    paths=list((ROOT/'src').rglob('*.py'))+[ROOT/'scripts/sim'/name for name in ['round14_action_sources.py','round14_logged_anchors.py','round12_diagnostics.py','round12_diagnostic_protocol.py','round11_closed_loop.py','round11_closed_loop_protocol.py','closed_loop.py']]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}

def make_manifest(checkpoint_root):
    design,rows=anchors.roster();recovery=json.loads((ROOT/'reports/ROUND12_CHECKPOINT_RECOVERY.json').read_text())
    known={r['path']:r['sha256'] for r in recovery['files']};models=[]
    for d in range(5):
        for arm in ['frozen','updated']:
            for seed in [70,71,72]:
                rel=f'round11b_data{d}_v1/r11b_d{d}_{arm}_r0_seed{seed}/checkpoint_best.pt'
                digest=sha(Path(checkpoint_root)/rel);assert digest==known['runs/results/'+rel]
                norm=f'artifacts/round11b_v1/data{d}_norm.json';split=f'artifacts/round11b_v1/data{d}_split.json'
                models.append(dict(dataset=d,arm=arm,seed=seed,path=rel,sha256=digest,normalizer=norm,normalizer_sha256=sha(ROOT/norm),split=split,split_sha256=sha(ROOT/split)))
    return dict(schema='round14-action-source-v1',protocol_sha256=sha(ROOT/'docs/ROUND14_ACTION_SOURCE_PROTOCOL.md'),source_sha256=sources(),rows=rows,models=models,horizon=50,noise_samples=8,candidates=64,anchor=100,planning_seed_base=510000,action_dtype='float32 for both physical and model evaluation')

def validate_manifest(m):
    assert m['schema']=='round14-action-source-v1' and m['rows']==anchors.roster()[1]
    assert m['source_sha256']==sources() and m['protocol_sha256']==sha(ROOT/'docs/ROUND14_ACTION_SOURCE_PROTOCOL.md')
    assert (m['horizon'],m['noise_samples'],m['candidates'],m['anchor'],m['planning_seed_base'])==(50,8,64,100,510000)
    assert [(x['dataset'],x['arm'],x['seed']) for x in m['models']]==[(d,a,s) for d in range(5) for a in ['frozen','updated'] for s in [70,71,72]]
    for x in m['models']:
        assert sha(ROOT/x['normalizer'])==x['normalizer_sha256'] and sha(ROOT/x['split'])==x['split_sha256']

def execute(m,index,checkpoint_root,data_root,device):
    validate_manifest(m);t0=time.monotonic();r=anchors.reconstruct(index,data_root);row=r['row'];snap=r['snapshot'];diag=anchors.diag
    acceleration=np.gradient(r['velocity'],.02,axis=0)
    nominal=diag.legacy.pd_plan(diag.QuadrotorSim(),snap['state'],r['trajectory'],r['velocity'],acceleration,100,50)
    bank=diag.TraceMPPI(None,n_samples=64,rng=np.random.default_rng(510000+index)).sample_bank(nominal)
    actions=np.concatenate([r['logged_actions'][None],bank['actions']],0).astype(np.float32)
    assert actions.shape==(65,50,5) and np.isfinite(actions).all()
    assert np.array_equal(actions[1],actions[2])
    noise=diag.noise_bundle([14,index,0],samples=8,horizon=50)
    truth=diag.simulate(snap,actions,noise,row['mean_wind'])
    for k in ['p','v','R','w','p_planner']:assert np.array_equal(truth[k][:,1],truth[k][:,2])
    traces={};diag.flatten_arrays('snapshot',snap,traces);diag.flatten_arrays('bank',bank,traces);diag.flatten_arrays('noise',noise,traces);diag.flatten_arrays('truth',truth,traces);diag.flatten_arrays('logged_truth',r['logged_truth'],traces);traces['actions']=actions
    actual=r['logged_truth'];actual_truth=dict(p=actual['p'][None,None],v=actual['state'][None,None,:,:3],w=actual['state'][None,None,:,9:],R=actual['R'][None,None],p_planner=(snap['state']['p']+.02*np.cumsum(actual['state'][:,:3],axis=0))[None,None])
    records=[]
    for spec in [x for x in m['models'] if x['dataset']==row['replicate']]:
        path=Path(checkpoint_root)/spec['path'];assert sha(path)==spec['sha256']
        model,state,art=load_model_from_checkpoint(path,torch.device(device))
        assert model.n_parameters()==39497
        pred,rotation=diag.predict(model,snap['history'],actions,torch.device(device));scales=art['normalizer'].metric_scales
        metrics={}
        for label,sl in [('logged',slice(0,1)),('candidate',slice(1,65))]:
            metrics[label]=diag.horizon_metrics(pred[sl],rotation[sl],{k:truth[k][:,sl] for k in ['p','v','R','w','p_planner']},snap,scales)
        metrics['actual_logged_future']=diag.horizon_metrics(pred[:1],rotation[:1],actual_truth,snap,scales)
        i=len(records);traces[f'model_{i}/state']=pred;traces[f'model_{i}/R']=rotation
        records.append(dict(identity=spec,metrics=metrics))
    return dict(status='complete',index=index,row=row,replay_errors=r['replay_errors'],csv_sha256=r['csv_sha256'],models=records,seconds=time.monotonic()-t0,device=str(device),gpu=torch.cuda.get_device_name() if str(device).startswith('cuda') else None,torch_version=torch.__version__,numpy_version=np.__version__,python=platform.python_version(),execution_source=os.environ.get('LATENT_WAM_SOURCE_COMMIT','local-development')),traces

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--write-manifest',action='store_true');ap.add_argument('--checkpoint-root',default='runs/results');ap.add_argument('--data-root',default='data/round11_sim_v1');ap.add_argument('--index',type=int);ap.add_argument('--output',type=Path);ap.add_argument('--device',default='cpu');args=ap.parse_args()
    if args.write_manifest:
        with args.manifest.open('x') as f:f.write(json.dumps(make_manifest(args.checkpoint_root),indent=2)+'\n')
        return
    if args.index is None or args.output is None:ap.error('index/output required')
    args.output.mkdir(parents=True,exist_ok=True);stem=args.output/f'anchor_{args.index:03d}'
    with stem.with_suffix('.claim').open('x') as f:f.write(str(os.getpid()))
    try:
        torch.set_num_threads(1)
        m=json.loads(args.manifest.read_text());result,traces=execute(m,args.index,args.checkpoint_root,args.data_root,args.device)
        np.savez_compressed(stem.with_suffix('.npz'),**traces)
        result.update(manifest_sha256=sha(args.manifest),trace_sha256=sha(stem.with_suffix('.npz')))
        with stem.with_suffix('.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ['models','row']},indent=2))
    except Exception:
        stem.with_suffix('.error').write_text(traceback.format_exc());raise

if __name__=='__main__':main()
