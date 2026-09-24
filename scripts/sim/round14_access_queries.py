"""Score all new access models on the exact accepted A1 banks and noise draws."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).resolve().parent))
import round12_diagnostics as diag
from latent_aero_wam.evaluation.evaluator import load_model_from_checkpoint

def sources():
 files=list((ROOT/'src').rglob('*.py'))+[ROOT/'scripts/sim'/n for n in ['round14_access_queries.py','round12_diagnostics.py','round12_diagnostic_protocol.py','round11_closed_loop.py','round11_closed_loop_protocol.py','closed_loop.py']]
 return {str(p.relative_to(ROOT)):diag.sha256(p) for p in files}
def make_manifest():
 control=json.loads((ROOT/'runs/evidence/round14_bundle/access_control.json').read_text());anchors=[]
 for i in range(20,40):
  path=ROOT/f'runs/results/round12a_formal_v1/a1_{i:04d}.json';r=json.loads(path.read_text());assert r['row']['wind'] in [6.1,8.5] and not r['pilot'];assert diag.sha256(path.with_suffix('.npz'))==r['trace_sha256']
  anchors.append(dict(original_index=i,result_sha256=diag.sha256(path),trace_sha256=r['trace_sha256']))
 return dict(schema='round14-access-query-v1',models=control['models'],anchors=anchors,source_sha256=sources(),protocol_sha256=diag.sha256(ROOT/'docs/ROUND14_ACCESS_EVALUATION.md'))
def validate(m):
 assert m['schema']=='round14-access-query-v1' and m['source_sha256']==sources() and m['protocol_sha256']==diag.sha256(ROOT/'docs/ROUND14_ACCESS_EVALUATION.md')
 assert [a['original_index'] for a in m['anchors']]==list(range(20,40))
 assert [(x['dataset'],x['arm'],x['seed']) for x in m['models']]==[(d,a,s) for d in range(5) for a in ['fixed','dynamic'] for s in [70,71,72]]
def execute(m,index,checkpoint_root,anchor_root,device):
 validate(m);start=time.monotonic();entry=m['anchors'][index];path=Path(anchor_root)/f"a1_{entry['original_index']:04d}.json";old=json.loads(path.read_text());assert diag.sha256(path)==entry['result_sha256'];npz=path.with_suffix('.npz');assert diag.sha256(npz)==entry['trace_sha256'];diag.validate_result(old,npz)
 with np.load(npz,allow_pickle=False) as z:
  prefix='anchor_100/';snapshot=dict(step=100,state={k:z[prefix+'snapshot/state/'+k] for k in ['p','v','R','w']},gust=z[prefix+'snapshot/gust'],history={k:z[prefix+'snapshot/history/'+k] for k in ['states','v_bodies','Rs','actions','deltas']},p_ref=z[prefix+'snapshot/p_ref'],v_ref=z[prefix+'snapshot/v_ref'])
  bank={k:z[prefix+'bank/'+k] for k in ['nominal','shifted_prev','perturbations','actions']}
  truth={k:z[prefix+'evaluation/'+k][:,:64] for k in ['p','v','R','w','p_planner']}
  checks=0
  for mi,rec in enumerate(old['anchors'][0]['models']):
   if not rec['model'].endswith('_r0'):continue
   scales=diag.Normalizer.load(ROOT/f"artifacts/round11b_v1/data{rec['dataset_replicate']}_norm.json").metric_scales
   metrics=diag.horizon_metrics(z[prefix+f'model_{mi}/state'],z[prefix+f'model_{mi}/R'],truth,snapshot,scales)
   for h in metrics:
    for key in ['original_E','physical_position_l2_m']:
     for stat in ['terminal','prefix_mean']:assert np.isclose(metrics[h][key][stat],rec['horizon_metrics'][h][key][stat],rtol=1e-10,atol=1e-12)
   checks+=1
  assert checks==30
  specs=[dict(model=x['config']['model']['name'],dataset_replicate=x['dataset'],model_seed=x['seed'],identity=x) for x in m['models']]
  def load(spec):
   x=spec['identity'];cp=Path(checkpoint_root)/x['path'];assert diag.sha256(cp)==x['checkpoint_sha256']
   model,state,art=load_model_from_checkpoint(cp,torch.device(device));assert state['config']==x['config'] and state['provenance']==x['provenance'] and model.n_parameters()==51785
   return model,art['normalizer'].metric_scales,dict(checkpoint_sha256=x['checkpoint_sha256'],arm=x['arm'])
  result,traces=diag.evaluate_anchor(snapshot,bank,specs,load,old['row'],mc_samples=8)
  for phase in ['construct','evaluate']:
   for k in ['gust','throttle','rate']:assert np.array_equal(traces[f'noise/{phase}/{k}'],z[prefix+f'noise/{phase}/{k}'])
  for k in ['p','v','R','w','p_planner']:assert np.allclose(traces['evaluation/'+k][:,:64],truth[k],rtol=1e-12,atol=1e-12)
  assert np.array_equal(traces['bank/actions'],z[prefix+'bank/actions'])
 return dict(status='complete',index=index,original_anchor=entry,row=old['row'],anchor=result,compatibility_checked_original_models=checks,seconds=time.monotonic()-start,device=device,torch_version=torch.__version__,gpu=torch.cuda.get_device_name() if device.startswith('cuda') else None),traces

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--write-manifest',action='store_true');ap.add_argument('--index',type=int);ap.add_argument('--output',type=Path);ap.add_argument('--checkpoint-root',default='runs/results');ap.add_argument('--anchor-root',default='runs/results/round12a_formal_v1');ap.add_argument('--device',default='cpu');a=ap.parse_args()
 if a.write_manifest:
  with a.manifest.open('x') as f:f.write(json.dumps(make_manifest(),indent=2)+'\n')
  return
 if a.index is None or a.output is None:ap.error('index/output required')
 a.output.mkdir(parents=True,exist_ok=True);p=a.output/f'anchor_{a.index:03d}'
 with p.with_suffix('.claim').open('x') as f:f.write(str(os.getpid()))
 try:
  torch.set_num_threads(1);r,z=execute(json.loads(a.manifest.read_text()),a.index,a.checkpoint_root,a.anchor_root,a.device)
  np.savez_compressed(p.with_suffix('.npz'),**z);r.update(trace_sha256=diag.sha256(p.with_suffix('.npz')),manifest_sha256=diag.sha256(a.manifest),execution_source=os.environ.get('LATENT_WAM_SOURCE_COMMIT','local-development'))
  with p.with_suffix('.json').open('x') as f:f.write(json.dumps(r,indent=2)+'\n')
  print(a.index,r['seconds'])
 except Exception:p.with_suffix('.error').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
