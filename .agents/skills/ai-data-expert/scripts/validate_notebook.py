#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
import pandas as pd

SCRIPT=Path(__file__).resolve();ROOT=SCRIPT.parents[4];CORE=ROOT/'data_expert';sys.path.insert(0,str(CORE))
from data_guard_v4 import analyze_dataframe


def semantic_checks(nb, data_path=None, target=None):
    code='\n'.join(c.source for c in nb.cells if c.cell_type=='code');markdown='\n'.join(c.source for c in nb.cells if c.cell_type=='markdown');checks=[]
    def add(name,ok,evidence=None,severity='ERROR'):checks.append({'name':name,'pass':bool(ok),'severity':severity,'evidence':evidence})
    add('no_traceback_output',not any('Traceback' in str(o) for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[])))
    add('test_isolation_structure','val_idx' in code and 'test_idx' in code and 'validation-selected' in code, 'validation/test indices and selection marker')
    if 'selected_threshold' in code:add('threshold_from_validation','predict_proba(X.iloc[val_idx])' in code or 'predict_proba(X_val)' in code,'threshold source')
    if 'classification_report' in code:add('per_class_report',True,'classification_report present')
    if re.search(r'production[- ]?ready',markdown,re.I):add('no_unsupported_production_claim',False,'production-ready phrase found',severity='WARNING')
    if data_path and target:
        df=pd.read_csv(data_path);guard=analyze_dataframe(df,target)
        if guard.get('target_missing_count',0):add('target_missing_separated','labeled_df=df[df[TARGET].notna()]' in code,guard.get('target_missing_count'))
        if guard.get('drop_feature_columns'):
            assignment=next((line for line in code.splitlines() if line.startswith('EXCLUDED_ID_FEATURES=')), '')
            add('identifier_proxy_excluded',all(repr(x) in assignment or f'"{x}"' in assignment for x in guard['drop_feature_columns']),{'detected':guard['drop_feature_columns'],'assignment':assignment})
        if guard.get('group_strategy'):
            add('group_aware_split','GroupShuffleSplit' in code and 'GROUP_OVERLAP' in code,guard.get('group_strategy'))
    errors=[c for c in checks if not c['pass'] and c['severity']=='ERROR'];warnings=[c for c in checks if not c['pass'] and c['severity']=='WARNING'];status='FAIL' if errors else ('REVIEW' if warnings else 'PASS')
    return {'status':status,'checks':checks,'errors':len(errors),'warnings':len(warnings)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('notebook');ap.add_argument('--timeout',type=int,default=300);ap.add_argument('--no-inplace',action='store_true');ap.add_argument('--data');ap.add_argument('--target');args=ap.parse_args();p=Path(args.notebook).resolve();nb=nbformat.read(p,as_version=4)
    os.environ.setdefault('PYTHONUTF8','1');os.environ.setdefault('PYTHONIOENCODING','utf-8');os.environ.setdefault('LOKY_MAX_CPU_COUNT','1')
    try:
        executed=NotebookClient(nb,timeout=args.timeout,kernel_name='python3',resources={'metadata':{'path':str(p.parent)}}).execute();
        if not args.no_inplace:nbformat.write(executed,p)
        sem=semantic_checks(executed,Path(args.data).resolve() if args.data else None,args.target);result={'status':'PASS' if sem['status']=='PASS' else sem['status'],'notebook':str(p),'cells':len(executed.cells),'semantic':sem};print(json.dumps(result,ensure_ascii=False));
        if sem['status']=='FAIL':raise RuntimeError('semantic notebook validation failed')
    except Exception as e:
        print(json.dumps({'status':'FAIL','notebook':str(p),'error':f'{type(e).__name__}: {e}'},ensure_ascii=False));raise
if __name__=='__main__':main()
