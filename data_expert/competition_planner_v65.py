from __future__ import annotations
from typing import Any
import pandas as pd
from competition_planner_v63 import CompetitionPlannerV63
from data_guard_v65 import analyze_dataframe

class CompetitionPlannerV65(CompetitionPlannerV63):
    def inspect(self,spec,df:pd.DataFrame|None=None)->dict[str,Any]:
        plan=super().inspect(spec,df)
        if df is None:
            return plan
        target=spec.target if spec.target in df.columns else None
        guard=analyze_dataframe(df,target=target)
        plan['data_guard']=guard
        plan['group_candidates']=list(dict.fromkeys([x.get('column') for x in guard.get('entity_candidates',[]) if x.get('column')]))
        plan['time_candidates']=[x.get('column') for x in guard.get('time_candidates',[]) if x.get('column')]
        tc=guard.get('time_candidates') or []
        bad=[x for x in tc if (not x.get('monotonic',True)) or x.get('non_monotonic_count',0)>0 or x.get('duplicate_count',0)>0 or x.get('irregular_interval_count',0)>0 or ((x.get('max_gap_ratio') or 1)>2.0)]
        plan['time_integrity']={
            'checked':bool(tc),
            'non_monotonic_detected':any(x.get('non_monotonic_count',0)>0 or not x.get('monotonic',True) for x in tc),
            'duplicate_timestamp_detected':any(x.get('duplicate_count',0)>0 for x in tc),
            'cadence_break_detected':any(x.get('irregular_interval_count',0)>0 or ((x.get('max_gap_ratio') or 1)>2.0) for x in tc),
            'candidates':tc,
            'status':'REVIEW' if bad else ('PASS' if tc else 'NOT_APPLICABLE')}
        req=str(plan.get('requested_validation') or '').lower(); risks={str(r).lower() for r in plan.get('risk_flags',[])}
        time_req=req in {'time-aware','chronological','rolling-origin'} or plan.get('category')=='timeseries' or any('time' in r for r in risks)
        group_req=req=='group-aware' or any(any(k in r for k in ['group','entity','driver','building','molecule','panel','household','family','crew','merchant','cohort']) for r in risks)
        groups=plan.get('group_candidates') or []
        if groups and tc and time_req and group_req:
            plan['inferred_validation']='temporal-group'; plan['validation_components']=['chronological','group-isolation']
        elif groups and req=='group-aware':
            plan['inferred_validation']='group-aware'; plan['validation_components']=['group-isolation']
        elif groups and req=='stratified' and group_req:
            plan['inferred_validation']='stratified'; plan['validation_components']=['stratified','group-sensitivity-check']
        plan['critical_leakage_columns']=[x['column'] for x in guard.get('target_leakage_proxies',[])]
        if plan['critical_leakage_columns']:
            plan.setdefault('unknowns',[]).append('near-direct target leakage feature(s) excluded before modeling')
        return plan
