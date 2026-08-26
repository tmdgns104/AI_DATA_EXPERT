#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT=Path(__file__).resolve(); ROOT=SCRIPT.parents[4]; CORE=ROOT/'data_expert'; sys.path.insert(0,str(CORE))
from enhanced_system import EnhancedSystem

def default(o):
    if isinstance(o,(np.integer,)):return int(o)
    if isinstance(o,(np.floating,)):return float(o)
    if isinstance(o,(np.bool_,)):return bool(o)
    if hasattr(o,'item'):
        try:return o.item()
        except Exception:pass
    return str(o)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv'); ap.add_argument('--task',required=True); ap.add_argument('--target'); ap.add_argument('--modality',default='tabular'); ap.add_argument('--timestamp-col'); ap.add_argument('--horizon'); ap.add_argument('--censor-col'); ap.add_argument('--entry-col'); ap.add_argument('--prediction-time'); ap.add_argument('--business-cost'); ap.add_argument('--domain-path',action='append',default=[]); ap.add_argument('--image-npz'); ap.add_argument('--monitoring',action='store_true'); ap.add_argument('--existing-model',action='store_true'); ap.add_argument('--deployment',action='store_true'); ap.add_argument('--streaming',action='store_true'); ap.add_argument('--size-gb',type=float); ap.add_argument('--out',required=True); args=ap.parse_args()
    profile={'modality':args.modality}; data_path=None
    if args.csv:
        p=Path(args.csv).resolve();
        if not p.exists():raise FileNotFoundError(p)
        df=pd.read_csv(p); data_path=str(p); profile.update(rows=len(df),columns=df.shape[1])
        if args.target:
            if args.target not in df.columns:raise ValueError(f'target not found: {args.target}')
            profile['target']=args.target; y=df[args.target]; profile['target_type']='continuous' if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True)>20 else 'categorical'
    if args.timestamp_col:profile['timestamp_col']=args.timestamp_col
    if args.horizon:profile['horizon']=args.horizon
    if args.censor_col:profile['censor_col']=args.censor_col
    if args.entry_col:profile['entry_col']=args.entry_col
    if args.prediction_time:profile['prediction_time']=args.prediction_time
    if args.business_cost:profile['business_cost']=args.business_cost
    if args.domain_path:profile['domain_paths']=[str(Path(x).resolve()) for x in args.domain_path]
    if args.image_npz:profile['image_npz']=str(Path(args.image_npz).resolve())
    if args.monitoring:profile['monitoring']=True
    if args.existing_model:profile['existing_model']=True
    if args.deployment:profile['deployment']=True
    if args.streaming:profile['streaming']=True
    if args.size_gb is not None:profile['size_gb']=args.size_gb
    problem={'id':'CODEX-USER-TASK','title':'Codex user task','task':args.task,'data_path':data_path,'profile':profile}; result=EnhancedSystem().run(problem); out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=default),encoding='utf-8')
    print(json.dumps({'primary_agent':result['routing'].get('primary_agent'),'execution_order':result['routing'].get('execution_order'),'intent':result['task_spec']['intent']['primary_intent'],'verification':result['verification']['status'],'challenger':result['challenger']['status'],'output':str(out.resolve())},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
