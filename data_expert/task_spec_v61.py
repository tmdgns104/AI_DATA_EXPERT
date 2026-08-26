from __future__ import annotations
from typing import Any
from task_spec_v5 import TaskSpecBuilderV5

class TaskSpecBuilderV61(TaskSpecBuilderV5):
    def build(self,problem:dict[str,Any])->dict[str,Any]:
        spec=super().build(problem); p=problem.get('profile',{}); modality=str(p.get('modality',spec.get('modality','tabular'))).lower()
        if modality!='time-series': return spec
        intent=spec.get('intent',{}); neg=set(intent.get('negated',[])); intents=set(intent.get('intents',[])); effective_forecast='FORECAST' in intents and 'FORECAST' not in neg
        horizon=p.get('horizon') or p.get('forecast_horizon')
        if not effective_forecast:
            spec['problem_type']='descriptive_time_series'
            spec['causal_or_predictive']='descriptive'
            spec['unknowns']=[u for u in spec.get('unknowns',[]) if u!='explicit forecast horizon']
            spec['assumptions']=[a for a in spec.get('assumptions',[]) if 'forecast horizon inferred' not in a]
            if spec.get('prediction_time','').startswith('next observation'): spec['prediction_time']='NOT_APPLICABLE'
        elif horizon:
            spec['prediction_time']=f'next {horizon}'
            spec['unknowns']=[u for u in spec.get('unknowns',[]) if u!='explicit forecast horizon']
            spec['assumptions']=[a for a in spec.get('assumptions',[]) if 'forecast horizon inferred' not in a]
        return spec
