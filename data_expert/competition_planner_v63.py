from __future__ import annotations
from typing import Any
import pandas as pd
from competition_planner_v62 import CompetitionPlannerV62


class CompetitionPlannerV63(CompetitionPlannerV62):
    def inspect(self, spec, df: pd.DataFrame | None = None) -> dict[str, Any]:
        plan=super().inspect(spec,df)
        guard=plan.get('data_guard') or {}
        tc=guard.get('time_candidates') or []
        if tc:
            bad=[x for x in tc if (not x.get('monotonic',True)) or x.get('non_monotonic_count',0)>0 or x.get('duplicate_count',0)>0]
            plan['time_integrity']={
                'checked':True,
                'non_monotonic_detected':any(x.get('non_monotonic_count',0)>0 or not x.get('monotonic',True) for x in tc),
                'duplicate_timestamp_detected':any(x.get('duplicate_count',0)>0 for x in tc),
                'candidates':tc,
                'status':'REVIEW' if bad else 'PASS',
            }
        else:
            plan['time_integrity']={'checked':False,'non_monotonic_detected':False,'duplicate_timestamp_detected':False,'candidates':[],'status':'NOT_APPLICABLE'}
        inferred=[x.get('column') for x in guard.get('entity_candidates',[]) if x.get('column')]
        if inferred:
            plan['group_candidates']=list(dict.fromkeys(list(plan.get('group_candidates') or [])+inferred))
            req=str(plan.get('requested_validation') or '').lower(); risks={str(r).lower() for r in plan.get('risk_flags',[])}
            time_req=req in {'time-aware','chronological','rolling-origin'} or plan.get('category')=='timeseries' or any('time' in r for r in risks)
            group_req=req=='group-aware' or any(any(k in r for k in ['group','entity','driver','building','molecule','panel']) for r in risks)
            if time_req and group_req and plan.get('time_candidates'):
                plan['inferred_validation']='temporal-group'; plan['validation_components']=['chronological','group-isolation']
            elif req=='group-aware':
                plan['inferred_validation']='group-aware'; plan['validation_components']=['group-isolation']
            elif req=='stratified' and group_req:
                plan['inferred_validation']='stratified'; plan['validation_components']=['stratified','group-sensitivity-check']
        return plan
