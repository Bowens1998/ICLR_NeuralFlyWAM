"""Full-window physical-position audit with accepted-error reproduction gate."""
import argparse,hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from latent_aero_wam.evaluation.evaluator import load_model_from_checkpoint,move_batch
from latent_aero_wam.evaluation.metrics import MetricScales,rollout_errors,aggregate,true_R_from_state
from latent_aero_wam.data.dataset import FlightStore,WindowDataset,make_loader

METRICS=['velocity','orientation','angular_rate','displacement','aggregate']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sources():
    return {str(p.relative_to(ROOT)):sha(p) for p in [*list((ROOT/'src').rglob('*.py')),Path(__file__).resolve()]}
def manifest(root):
    recovery={r['path']:r['sha256'] for r in json.loads((ROOT/'reports/ROUND12_CHECKPOINT_RECOVERY.json').read_text())['files']}
    old_eval=json.loads((ROOT/'reports/ROUND14_OFFLINE_COMPONENTS.json').read_text())['source_sha256'];models=[]
    for d in range(5):
        accepted=json.loads((ROOT/f'runs/evidence/round14_bundle/round14_access_data{d}_acceptance.json').read_text());assert accepted['accepted_runs']==6
        for arm in ['frozen','updated','fixed','dynamic']:
            for seed in [70,71,72]:
                folder=(f'round11b_data{d}_v1/r11b_d{d}_{arm}_r0_seed{seed}' if arm in ['frozen','updated'] else f'round14_access_data{d}_v1/r14_d{d}_{arm}_seed{seed}')
                cp=folder+'/checkpoint_best.pt';ev=folder+'/eval/d2_static_ood_per_window.npz'
                ch=sha(Path(root)/cp);eh=sha(Path(root)/ev)
                assert ch==(recovery['runs/results/'+cp] if arm in ['frozen','updated'] else accepted['sha256']['runs/results/'+cp])
                assert eh==(old_eval['runs/results/'+ev] if arm in ['frozen','updated'] else accepted['sha256']['runs/results/'+ev])
                models.append(dict(dataset=d,arm=arm,seed=seed,checkpoint=cp,checkpoint_sha256=ch,evaluation=ev,evaluation_sha256=eh))
    return dict(schema='round14-position-v1',models=models,source_sha256=sources(),protocol_sha256=sha(ROOT/'docs/ROUND14_POSITION_PROTOCOL.md'),rtol=5e-4,atol=1e-6,batch_size=256)
def validate(m):
    assert m['schema']=='round14-position-v1' and m['source_sha256']==sources() and m['protocol_sha256']==sha(ROOT/'docs/ROUND14_POSITION_PROTOCOL.md')
    assert (m['rtol'],m['atol'],m['batch_size'])==(5e-4,1e-6,256)
    assert [(x['dataset'],x['arm'],x['seed']) for x in m['models']]==[(d,a,s) for d in range(5) for a in ['frozen','updated','fixed','dynamic'] for s in [70,71,72]]

@torch.no_grad()
def execute(m,index,root,device):
    validate(m);start=time.monotonic();spec=m['models'][index];cp=Path(root)/spec['checkpoint'];ep=Path(root)/spec['evaluation']
    assert sha(cp)==spec['checkpoint_sha256'] and sha(ep)==spec['evaluation_sha256']
    model,_,art=load_model_from_checkpoint(cp,torch.device(device));store=FlightStore(art['manifest']);ds=WindowDataset(art['manifest'],'d2_static_ood',art['normalizer'],store,stride=1)
    assert len(ds)==85500 and ds.H==50 and ds.K==100
    scales=MetricScales(art['normalizer'].metric_scales,device=device);chunks={k:[] for k in METRICS};pos=[];euler=[];ids=[];centres=[]
    names=ds.flight_names
    for batch in make_loader(ds,256,False,0,pin_memory=device.startswith('cuda')):
        fid=batch['flight_id'].numpy();t=batch['centre_index'].numpy()
        p0=np.stack([store.get(names[int(f)]).p[int(c)] for f,c in zip(fid,t)])
        true_p=np.stack([store.get(names[int(f)]).p[int(c)+1:int(c)+51] for f,c in zip(fid,t)])
        b=move_batch(batch,torch.device(device));out=model(b)
        errs=rollout_errors(out.state.float(),out.R.float(),b['future_state'],true_R_from_state(b['future_state']),scales,.02);errs['aggregate']=aggregate(errs)
        for k,v in errs.items():chunks[k].append(v.cpu().numpy())
        phat=p0[:,None]+torch.cumsum(out.state[...,:3].float()*.02,dim=1).cpu().numpy()
        peuler=p0[:,None]+torch.cumsum(b['future_state'][...,:3]*.02,dim=1).cpu().numpy()
        pos.append(np.linalg.norm(phat-true_p,axis=-1));euler.append(np.linalg.norm(phat-peuler,axis=-1));ids.append(fid);centres.append(t)
    errors={k:np.concatenate(v) for k,v in chunks.items()};ids=np.concatenate(ids);centres=np.concatenate(centres);pos=np.concatenate(pos);euler=np.concatenate(euler)
    checks={};records=[]
    with np.load(ep,allow_pickle=False) as old:
        assert np.array_equal(ids,old['flight_ids']) and np.array_equal(centres,old['centre_index']) and np.array_equal(names,old['flight_names'])
        for k,a in errors.items():
            expected=old['err_'+k];assert a.shape==expected.shape and np.isfinite(a).all()
            checks[k]=dict(max_per_window_abs_difference=float(np.max(np.abs(a-expected))))
            for fid in range(30):
                mask=ids==fid
                assert np.allclose(a[mask].mean(0),expected[mask].mean(0),rtol=m['rtol'],atol=m['atol']),(k,fid,'horizon')
                assert np.isclose(a[mask].mean(),expected[mask].mean(),rtol=m['rtol'],atol=m['atol']),(k,fid,'mean')
        for fid,name in enumerate(names):
            if not str(name).endswith(('_50wind','_70wind')):continue
            mask=ids==fid;assert mask.sum()==2850
            metrics={k:dict(prefix_mean=float(a[mask].mean()),terminal={str(h):float(a[mask,h-1].mean()) for h in [1,5,10,25,50]}) for k,a in {'physical_position_l2_m':pos,'planner_position_l2_m':euler}.items()}
            records.append(dict(flight=str(name),wind=6.1 if str(name).endswith('_50wind') else 8.5,metrics=metrics))
    assert len(records)==20 and np.isfinite(pos).all() and np.isfinite(euler).all()
    mask=np.isin(ids,[i for i,n in enumerate(names) if str(n).endswith(('_50wind','_70wind'))]);assert mask.sum()==57000
    trace=dict(physical_position_l2_m=pos[mask],planner_position_l2_m=euler[mask],flight_ids=ids[mask],centre_index=centres[mask],flight_names=np.asarray(names))
    return dict(status='complete',index=index,identity=spec,checks=checks,flights=records,seconds=time.monotonic()-start,device=device,gpu=torch.cuda.get_device_name() if device.startswith('cuda') else None,torch_version=torch.__version__),trace

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--write-manifest',action='store_true');ap.add_argument('--root',default='runs/results');ap.add_argument('--index',type=int);ap.add_argument('--output',type=Path);ap.add_argument('--device',default='cuda');a=ap.parse_args()
    if a.write_manifest:
        with a.manifest.open('x') as f:f.write(json.dumps(manifest(a.root),indent=2)+'\n')
        return
    if a.index is None or a.output is None:ap.error('index/output required')
    a.output.mkdir(parents=True,exist_ok=True);p=a.output/f'model_{a.index:03d}'
    with p.with_suffix('.claim').open('x') as f:f.write(str(os.getpid()))
    try:
        torch.set_num_threads(1);m=json.loads(a.manifest.read_text());result,trace=execute(m,a.index,a.root,a.device)
        np.savez_compressed(p.with_suffix('.npz'),**trace);result.update(manifest_sha256=sha(a.manifest),trace_sha256=sha(p.with_suffix('.npz')),execution_source=os.environ.get('LATENT_WAM_SOURCE_COMMIT','local-development'))
        with p.with_suffix('.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
        print(a.index,result['seconds'])
    except Exception:p.with_suffix('.error').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
