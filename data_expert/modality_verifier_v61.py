from __future__ import annotations
from typing import Any
from v4_system import RuntimeVerifierV4

class ModalityVerifierV61:
    def __init__(self): self.base=RuntimeVerifierV4()
    def verify(self,problem:dict[str,Any],result:dict[str,Any]):
        base=self.base.verify(problem,result); checks=list(base.get('checks',[])); task_spec=result.get('task_spec',{}); modality=str(problem.get('profile',{}).get('modality','tabular')).lower(); intent=task_spec.get('intent',{}); neg=set(intent.get('negated',[])); intents=set(intent.get('intents',[])); effective_forecast='FORECAST' in intents and 'FORECAST' not in neg
        if modality=='time-series' and effective_forecast:
            checks=[c for c in checks if c['name'] not in {'dl_routed','actual_dl_execution','dl_small_batch_sanity'}]
            ts=next((o for o in result.get('expert_outputs',[]) if o.get('agent')=='time-series'),{}); markers=set(ts.get('markers',[])); decision=ts.get('DECIDE',{})
            def add(name,ok,evidence=None,severity='ERROR'): checks.append({'name':name,'pass':bool(ok),'severity':severity,'evidence':evidence})
            add('timeseries_specialist_routed',result.get('routing',{}).get('primary_agent')=='time-series',result.get('routing',{}).get('execution_order'))
            add('timestamp_integrity_checked','timestamp_integrity_check' in markers,decision.get('timestamp_repair'))
            add('chronological_split','temporal_split' in markers,sorted(markers)); add('train_only_scaling','train_only_scaling' in markers,sorted(markers)); add('naive_baseline_present','naive_baseline' in markers,sorted(markers)); add('actual_rnn_execution','actual_rnn_execution' in markers,sorted(markers)); add('rnn_lstm_compared','rnn_lstm_comparison' in markers,decision.get('test_metrics')); add('final_test_isolated','final_holdout_once' in markers,sorted(markers)); add('argument_ledger_present',bool(result.get('argument_ledger',{}).get('nodes')),result.get('argument_ledger',{}).get('nodes')); add('shared_evidence_present',result.get('shared_evidence',{}).get('record_count',0)>0,result.get('shared_evidence',{}).get('record_count'))
            if 'explicit forecast horizon' in task_spec.get('unknowns',[]): add('forecast_horizon_confirmed',False,'forecast horizon unresolved',severity='WARNING')
        errors=[c for c in checks if not c['pass'] and c['severity']=='ERROR']; warnings=[c for c in checks if not c['pass'] and c['severity']=='WARNING']
        return {'status':'FAIL' if errors else ('REVIEW' if warnings else 'PASS'),'passed':sum(bool(c['pass']) for c in checks),'total':len(checks),'score_pct':100*sum(bool(c['pass']) for c in checks)/len(checks) if checks else 100.0,'errors':len(errors),'warnings':len(warnings),'checks':checks}
