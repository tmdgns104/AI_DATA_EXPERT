from __future__ import annotations
from typing import Any
import re
import numpy as np, pandas as pd
from data_guard_v4 import analyze_dataframe, split_labeled_unlabeled

GROUP_HINT=re.compile(r'(driver|patient|subject|user|customer|store|shop|building|household|molecule|group|batch|lot|device|machine|equipment|product|item)[_-]?(id|code|no|number)?$',re.I)
TIME_HINT=re.compile(r'(date|time|timestamp|datetime)',re.I)
PROB_METRICS={'roc_auc','roc_auc_macro','log_loss','gini'}
COMPLEX_METRICS={'wrmsse','pinball'}

class CompetitionPlannerV6:
    def inspect(self,spec,df:pd.DataFrame|None=None)->dict[str,Any]:
        plan={'metric':spec.metric,'direction':spec.direction,'category':spec.category,'target':spec.target,
              'requested_validation':spec.validation,'risk_flags':list(spec.risk_flags),'unknowns':[]}
        guard=None
        if df is not None:
            target=spec.target if spec.target in df.columns else None
            guard=analyze_dataframe(df,target=target)
            plan['data_guard']=guard
            cols=list(map(str,df.columns)); lower={c.lower():c for c in cols}
            group_cols=[c for c in cols if GROUP_HINT.search(c.lower())]
            time_cols=[c for c in cols if TIME_HINT.search(c.lower())]
            plan['group_candidates']=group_cols
            plan['time_candidates']=time_cols
            if target:
                labeled,unlabeled=split_labeled_unlabeled(df,target); plan['labeled_rows']=len(labeled); plan['unlabeled_rows']=len(unlabeled)
                y=labeled[target]
                if spec.category=='classification' and y.nunique(dropna=True)<=30:
                    vc=y.value_counts(dropna=True); plan['minority_support']=int(vc.min()) if len(vc) else 0
                    plan['rare_event_reliability']='VERY_LOW' if len(vc) and vc.min()<10 else ('LOW' if len(vc) and vc.min()<30 else 'OK')
        if spec.category=='timeseries': inferred='rolling-origin' if spec.validation=='rolling-origin' else 'chronological'
        elif df is not None and (plan.get('group_candidates') or (guard and guard.get('group_strategy'))) and (spec.validation=='group-aware' or any('group' in r or 'leakage' in r for r in spec.risk_flags)):
            inferred='group-aware'
        elif df is not None and plan.get('time_candidates') and spec.validation in {'time-aware','chronological','rolling-origin'}:
            inferred='chronological'
        else: inferred='stratified' if spec.category in {'classification','vision'} else 'kfold'
        plan['inferred_validation']=inferred
        plan['submission_mode']='probability' if spec.metric in PROB_METRICS else ('class_label' if spec.category in {'classification','vision'} else 'continuous')
        plan['metric_runtime']='EXACT_PROXY' if spec.metric not in COMPLEX_METRICS else 'SPEC_KNOWN_RUNTIME_APPROX'
        if spec.metric in COMPLEX_METRICS: plan['unknowns'].append('exact competition weighting/hierarchy requires original competition artifacts')
        return plan
