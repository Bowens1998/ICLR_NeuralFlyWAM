"""Independent access-control manifest; unchanged original rollout and scoring."""
import argparse,hashlib,json,os,re,sys,traceback
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).resolve().parent))
import round11_closed_loop as engine
from latent_aero_wam.evaluation.evaluator import load_model_from_checkpoint
from latent_aero_wam.utils.config import config_hash

TRAIN_SOURCE='413b5c105739d7c468412b4ff49f71f3ba5e0d1b'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source_hashes():
    files=list((ROOT/'src').rglob('*.py'))+[ROOT/'scripts/sim'/n for n in ['round14_access_control.py','round11_closed_loop.py','round11_closed_loop_protocol.py','closed_loop.py']]
    return {str(p.relative_to(ROOT)):sha(p) for p in files}
def rows():
    out=[]
    for d in range(5):
        for arm in ['fixed','dynamic']:
            for seed in [70,71,72]:
                for wind in [6.1,8.5]:
                    for e in range(10):out.append(dict(index=len(out),kind='model',dataset_replicate=d,model=f'r14_d{d}_{arm}',arm=arm,model_seed=seed,wind=wind,episode=e,environment_seed=310000+e,trajectory_seed=320000+e,planning_seed=330000+e))
    return out

def make_manifest(root):
    models=[];acceptance={}
    for d in range(5):
        ap=ROOT/f'runs/evidence/round14_bundle/round14_access_data{d}_acceptance.json';a=json.loads(ap.read_text());assert a['accepted_runs']==6 and a['accepted_npz']==12
        acceptance[str(ap.relative_to(ROOT))]=sha(ap)
        for arm in ['fixed','dynamic']:
            for seed in [70,71,72]:
                rel=f'round14_access_data{d}_v1/r14_d{d}_{arm}_seed{seed}/checkpoint_best.pt';p=Path(root)/rel
                assert sha(p)==a['sha256']['runs/results/'+rel]
                cp=torch.load(p,map_location='cpu',weights_only=False);cfg=cp['config'];prov=cp['provenance']
                assert prov['git_commit']==TRAIN_SOURCE and prov['config_hash']==config_hash(cfg)
                models.append(dict(dataset=d,arm=arm,seed=seed,path=rel,checkpoint_sha256=sha(p),config=cfg,provenance=prov,split_sha256=sha(ROOT/cfg['data']['manifest_path']),normalizer_sha256=sha(ROOT/cfg['data']['normalizer_path'])))
    rr=rows();pilot=[r['index'] for r in rr if r['dataset_replicate']==0 and r['model_seed']==70 and r['episode']==0]
    assert pilot==[0,10,60,70]
    return dict(schema='round14-access-control-v1',source_sha256=source_hashes(),protocol_sha256=sha(ROOT/'docs/ROUND14_ACCESS_EVALUATION.md'),training_acceptance=acceptance,models=models,rows=rr,pilot_indices=pilot,scored_steps=1500)

def validate(m):
    assert m['schema']=='round14-access-control-v1' and m['source_sha256']==source_hashes()
    assert m['protocol_sha256']==sha(ROOT/'docs/ROUND14_ACCESS_EVALUATION.md')
    assert m['rows']==rows() and m['pilot_indices']==[0,10,60,70] and m['scored_steps']==1500
    assert [(x['dataset'],x['arm'],x['seed']) for x in m['models']]==[(d,a,s) for d in range(5) for a in ['fixed','dynamic'] for s in [70,71,72]]
    for x in m['models']:
        assert sha(ROOT/x['config']['data']['manifest_path'])==x['split_sha256']
        assert sha(ROOT/x['config']['data']['normalizer_path'])==x['normalizer_sha256']

def execute(m,index,root,device):
    validate(m);row=m['rows'][index];spec=next(x for x in m['models'] if (x['dataset'],x['arm'],x['seed'])==(row['dataset_replicate'],row['arm'],row['model_seed']))
    path=Path(root)/spec['path'];assert sha(path)==spec['checkpoint_sha256']
    model,cp,_=load_model_from_checkpoint(path,torch.device('cpu'))
    assert cp['config']==spec['config'] and cp['provenance']==spec['provenance'] and model.n_parameters()==51785
    assert all(torch.isfinite(v).all() for v in cp['model'].values())
    del model,cp
    result=engine.run(row,path,device,scored_steps=1500)
    result.update(schema=m['schema'],checkpoint_sha256=spec['checkpoint_sha256'],source_sha256=m['source_sha256'],gpu=torch.cuda.get_device_name() if device.startswith('cuda') else None)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--write-manifest',action='store_true');ap.add_argument('--checkpoint-root',default='runs/results');ap.add_argument('--index',type=int);ap.add_argument('--pilot',action='store_true');ap.add_argument('--output',type=Path);ap.add_argument('--device',default='cuda');args=ap.parse_args()
    if args.write_manifest:
        with args.manifest.open('x') as f:f.write(json.dumps(make_manifest(args.checkpoint_root),indent=2)+'\n')
        return
    if args.index is None or args.output is None:ap.error('index/output required')
    source=os.environ.get('LATENT_WAM_SOURCE_COMMIT','');assert re.fullmatch('[0-9a-f]{40}',source)
    m=json.loads(args.manifest.read_text());validate(m)
    index=m['pilot_indices'][args.index] if args.pilot else args.index
    args.output.mkdir(parents=True,exist_ok=True);path=args.output/f'episode_{index:04d}.json'
    identity=dict(index=index,manifest_sha256=sha(args.manifest),execution_source=source,pilot=args.pilot)
    with path.with_suffix('.claim').open('x') as f:f.write(json.dumps(identity))
    try:
        torch.set_num_threads(1)
        result=execute(m,index,args.checkpoint_root,args.device);result.update(identity)
        with path.open('x') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(index,result['failed'],result['elapsed_seconds'])
    except Exception:
        path.with_suffix('.error').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
