"""Reconstruct complete original logged anchors before action interventions."""
import hashlib
import json
import sys
from ast import literal_eval
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import round12_diagnostics as diag
from latent_aero_wam.data.parser import load_flight
from latent_aero_wam.utils.rotation import matrix_to_quat_np
from latent_aero_wam.sim import QuadrotorSim, SimParams, random_trajectory

ATOL=5.1e-9

def roster():
    design=json.loads((ROOT/'configs/sim/round11_generation.json').read_text())
    rows=[r for r in design['rows'] if r['group']=='static' and r['mean_wind'] in (6.1,8.5)]
    assert len(rows)==100
    return design,rows

def reconstruct(index,data_root):
    design,rows=roster();row=rows[index];path=Path(data_root)/(row['name']+'.csv')
    generation=json.loads(path.with_suffix('.generation.json').read_text())
    digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==generation['csv_sha256']
    assert row==generation['row'] and design['params']==generation['params']
    raw=pd.read_csv(path,nrows=151)
    pp=SimParams(**design['params']);sim=QuadrotorSim(pp,seed=row['noise_seed'])
    trajectory=random_trajectory(np.random.default_rng(row['trajectory_seed']),60.,pp.dt)
    assert hashlib.sha256(trajectory.tobytes()).hexdigest()==generation['trajectory_sha256']
    velocity=np.gradient(trajectory,pp.dt,axis=0)
    state=dict(p=trajectory[0].copy(),v=np.zeros(3),R=np.eye(3),w=np.zeros(3));gust=np.zeros(3)
    errors={k:0. for k in ['p','v','R','w','T_sp','q_sp']};snapshot=None
    for i in range(151):
        if i==100:
            snapshot=dict(step=i,state={k:v.copy() for k,v in state.items()},gust=gust.copy(),
                          p_ref=trajectory[101:151].copy(),v_ref=velocity[101:151].copy())
        gust=gust+pp.dt*(-gust/pp.gust_tau)+sim.rng.normal(0,pp.gust_std*max(row['mean_wind'],.3)*np.sqrt(2*pp.dt/pp.gust_tau),3)
        throttle,R_sp=sim.baseline_controller(state,trajectory[i],velocity[i])
        for k in errors:
            actual=(np.asarray([throttle]) if k=='T_sp' else matrix_to_quat_np(R_sp[None])[0] if k=='q_sp' else state[k])
            saved=np.asarray(literal_eval(raw.iloc[i][k]))
            errors[k]=max(errors[k],float(np.max(np.abs(actual-saved))))
        if i<150:state=sim.step(state,throttle,R_sp,np.array([row['mean_wind'],0.,0.])+gust)
    if max(errors.values())>ATOL:raise ValueError(f'original CSV replay mismatch: {errors}')
    flight=load_flight(path)
    snapshot['history']=dict(states=flight.state[:101].copy(),v_bodies=flight.v_body[:101].copy(),
                             Rs=flight.R[:101].copy(),actions=flight.action[:100].copy(),deltas=flight.delta_state[:100].copy())
    return dict(row=row,snapshot=snapshot,logged_actions=flight.action[100:150].copy(),
                logged_truth=dict(p=flight.p[101:151].copy(),state=flight.state[101:151].copy(),R=flight.R[101:151].copy()),
                trajectory=trajectory,velocity=velocity,replay_errors=errors,csv_sha256=digest)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--indices',nargs='+',type=int,default=[0,1]);ap.add_argument('--data-root',default='data/round11_sim_v1');args=ap.parse_args()
    for index in args.indices:
        result=reconstruct(index,args.data_root)
        print(json.dumps(dict(index=index,name=result['row']['name'],replay_errors=result['replay_errors'],csv_sha256=result['csv_sha256'])),flush=True)
