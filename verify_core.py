import hashlib,json
from pathlib import Path
import torch
from latent_aero_wam.data.normalize import Normalizer
from latent_aero_wam.models import build_model
root=Path(__file__).resolve().parent
manifest=json.loads((root/'PACKAGE_MANIFEST.json').read_text())
for name,sha in manifest['files_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
entries=json.loads((root/'CHECKPOINT_EXPORTS.json').read_text())
assert len(entries)==450
for e in entries:
    norm=Normalizer.load(root/e['config']['data']['normalizer_path'])
    model=build_model(e['config']['model'],norm,.02)
    state=torch.load(root/e['export_path'],map_location='cpu',weights_only=True)['model']
    model.load_state_dict(state,strict=True)
    assert sum(p.numel() for p in model.parameters())==(51785 if e['config']['model']['family']=='context_access_control' else 39497)
print('All packaged hashes and450 exact model loads verified.')
fresh=root/'reports/ROUND13_FRESH_RESULTS.json'
if fresh.exists():
    import numpy as np
    from scipy.stats import t
    for wind,v in json.loads(fresh.read_text())['wind_strata'].items():
        s=v['per_arm_dataset_seed_episode_scores']
        f,u=np.array(s['frozen']),np.array(s['updated'])
        assert f.shape==u.shape==(5,3,20)
        delta=(u-f).mean(axis=1)
        assert np.allclose(delta,v['dataset_by_episode'])
        assert np.isclose(f.mean(),v['frozen_mean']) and np.isclose(u.mean(),v['updated_mean'])
        for axis,key in [(1,'dataset_contrast'),(0,'episode_contrast_conditional_on_trained_roster')]:
            d=delta.mean(axis=axis);mean=d.mean();half=t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
            assert np.allclose(d,v[key]['values'])
            assert np.allclose([mean-half,mean+half],v[key]['descriptive_t95ci'])
    print('Fresh-panel means, paired matrices and both uncertainty axes reproduced.')
# Recompute new uncertainty summaries from the retained five-dataset differences.
import numpy as np
from scipy.stats import t
count=0
def check(node):
    global count
    if isinstance(node,dict):
        if 'dataset_differences' in node and 'descriptive_t95ci' in node:
            d=np.array(node['dataset_differences']);assert d.shape==(5,)
            mean=d.mean();half=t.ppf(.975,4)*d.std(ddof=1)/np.sqrt(5)
            assert np.isclose(mean,node['mean'],rtol=1e-10,atol=1e-12)
            assert np.allclose([mean-half,mean+half],node['descriptive_t95ci'],rtol=1e-8,atol=1e-10)
            count+=1
        for v in node.values():check(v)
    elif isinstance(node,list):
        for v in node:check(v)
for path in (root/'reports').glob('ROUND1[45]_*.json'):
    j=json.loads(path.read_text());check(j.get('summary',{}))
print(f'Round14/15: {count} five-dataset mean/interval summaries reproduced.')
import sys
sys.path.insert(0,str(root/'scripts/analysis'))
from round15_results import factorial
for panel in ['QUERY','CONTROL']:
    result=json.loads((root/f'reports/ROUND15_{panel}_RESULTS.json').read_text())
    assert len(result['records'])==1800
    for metric,expected in result['summary'].items():
        assert factorial(result['records'],metric)==expected
print('Round15 query/control factorials reproduced from all3600 retained model/episode scores.')
from round16_verify_compact import verify
for panel in ['OFFLINE','QUERY','CONTROL']:
    result=json.loads((root/f'reports/ROUND16_{panel}_RESULTS.json').read_text())
    print('Round16 independent compact-record verification:',verify(result))
from round17_verify_compact import verify as verify17
reference=json.loads((root/'reports/ROUND17_RESULTS.json').read_text())
learned=json.loads((root/'reports/ROUND16_CONTROL_RESULTS.json').read_text())
print('Round17 independent compact-record verification:',verify17(reference,learned))
from round18_verify_compact import verify as verify18
for panel in ['OFFLINE','QUERY','CONTROL']:
    result=json.loads((root/f'reports/ROUND18_{panel}_RESULTS.json').read_text())
    print('Round18 independent compact-record verification:',verify18(result))
from round18_assess import classify
control=json.loads((root/'reports/ROUND18_CONTROL_RESULTS.json').read_text())
interpretation=json.loads((root/'reports/ROUND18_INTERPRETATION.json').read_text())
for key,value in classify(control['summary']['rmse_m']).items():
    assert interpretation[key] == value,key
print('Prespecified confirmation interpretation reconstructed without outcome selection.')
