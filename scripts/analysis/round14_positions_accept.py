"""Verify physical-position arrays, full window identity and per-flight summaries."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/sim'))
import round14_positions as p

def accept(path,manifest,source):
    m=json.loads(Path(manifest).read_text());p.validate(m);r=json.loads(Path(path).read_text());spec=m['models'][r['index']]
    assert r['status']=='complete' and r['identity']==spec and r['manifest_sha256']==p.sha(manifest) and r['execution_source']==source
    assert set(r['checks'])==set(p.METRICS)
    assert all(np.isfinite(v['max_per_window_abs_difference']) and v['max_per_window_abs_difference']>=0 for v in r['checks'].values())
    trace=Path(path).with_suffix('.npz');assert p.sha(trace)==r['trace_sha256']
    old=ROOT/'runs/results'/spec['evaluation'];assert p.sha(old)==spec['evaluation_sha256']
    with np.load(trace,allow_pickle=False) as z,np.load(old,allow_pickle=False) as original:
        names=original['flight_names'];ids=original['flight_ids'];centres=original['centre_index']
        selected=[i for i,n in enumerate(names) if str(n).endswith(('_50wind','_70wind'))];mask=np.isin(ids,selected)
        assert mask.sum()==57000 and len(selected)==20
        assert np.array_equal(z['flight_ids'],ids[mask]) and np.array_equal(z['centre_index'],centres[mask]) and np.array_equal(z['flight_names'],names)
        assert len(r['flights'])==20 and [v['flight'] for v in r['flights']]==[str(names[i]) for i in selected]
        for metric in ['physical_position_l2_m','planner_position_l2_m']:
            a=z[metric];assert a.shape==(57000,50) and np.isfinite(a).all() and (a>=0).all()
            for fid,record in zip(selected,r['flights']):
                fm=z['flight_ids']==fid;assert fm.sum()==2850
                expected=record['metrics'][metric]
                assert np.isclose(a[fm].mean(),expected['prefix_mean'],rtol=1e-10,atol=1e-12)
                for h in [1,5,10,25,50]:assert np.isclose(a[fm,h-1].mean(),expected['terminal'][str(h)],rtol=1e-10,atol=1e-12)
    return dict(index=r['index'],result_sha256=p.sha(path),trace_sha256=r['trace_sha256'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--source',required=True);ap.add_argument('--pilot',action='store_true');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    indices=[0,6] if a.pilot else range(60);records=[accept(a.root/f'model_{i:03d}.json',a.manifest,a.source) for i in indices]
    with a.output.open('x') as f:f.write(json.dumps(dict(accepted_models=len(records),pilot=a.pilot,records=records,manifest_sha256=p.sha(a.manifest),source=a.source),indent=2)+'\n')
    print('accepted models',len(records))
