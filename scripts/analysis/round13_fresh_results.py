"""Analyze the complete locked fresh panel; preserve dataset and episode axes."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
import round12_controls_accept as acceptance


def describe(a):
    a=np.asarray(a,dtype=float)
    if a.ndim!=1 or not np.isfinite(a).all() or len(a)<2:raise ValueError('invalid contrast')
    mean=float(a.mean());half=float(t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)))
    return dict(values=a.tolist(),mean=mean,descriptive_t95ci=[mean-half,mean+half],positive=int((a>0).sum()),n=len(a))


def analyze(roster, records):
    if roster.get('evaluation_panel')!='round13_fresh_v1' or roster['phase']!='formal':raise ValueError('wrong panel')
    scores=np.full((2,5,3,20,2),np.nan);seen=set();failures=[]
    for record in records:
        r=record['row'];m=roster['checkpoint_manifest']['checkpoints'][r['checkpoint_index']]
        key=([6.1,8.5].index(r['wind']),r['dataset_replicate'],[70,71,72].index(r['model_seed']),r['episode'],int(m['update_context']))
        if key in seen:raise ValueError('duplicate paired outcome')
        seen.add(key);scores[key]=record['capped_tracking_rmse_m']
        if record['failed']:failures.append(dict(index=r['index'],reason=record['failure_reason'],score=record['capped_tracking_rmse_m']))
    if len(seen)!=1200 or not np.isfinite(scores).all():raise ValueError('incomplete panel')
    out={}
    for i,w in enumerate(['6.1','8.5']):
        delta=scores[i,...,1]-scores[i,...,0] # dataset, seed, episode
        matrix=delta.mean(axis=1)
        out[w]=dict(frozen_mean=float(scores[i,...,0].mean()),updated_mean=float(scores[i,...,1].mean()),
                    dataset_contrast=describe(matrix.mean(axis=1)),
                    episode_contrast_conditional_on_trained_roster=describe(matrix.mean(axis=0)),
                    dataset_by_episode=matrix.tolist(),dataset_by_seed=delta.mean(axis=2).tolist(),
                    per_arm_dataset_seed_episode_scores={a:scores[i,...,j].tolist() for j,a in enumerate(['frozen','updated'])})
    return dict(wind_strata=out,retained_failures=failures,accepted_episodes=1200,
                uncertainty='Five training datasets, averaging3seeds20episodes; episode intervals conditional on fixed trained roster. Matrix cells are not independent population replicates.',
                post_review_extension=True,no_new_p_values=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['protocol','root','acceptance','output']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();roster,records,accepted=acceptance.load_accepted(args.protocol,args.root,args.acceptance)
    out=analyze(roster,records);out.update(protocol_sha256=acceptance.protocol.file_hash(args.protocol),acceptance_sha256=acceptance.protocol.file_hash(args.acceptance),source_commit=accepted['source_commit'])
    args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({w:{k:v[k] for k in ['frozen_mean','updated_mean','dataset_contrast','episode_contrast_conditional_on_trained_roster']} for w,v in out['wind_strata'].items()},indent=2))
if __name__=='__main__':main()
