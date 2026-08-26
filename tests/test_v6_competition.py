import sys, unittest
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'data_expert'))
from v6_system import V6System
from competition_spec_v6 import CompetitionSpecBuilderV6
from competition_planner_v6 import CompetitionPlannerV6
from output_renderer_v6 import HumanFriendlyRendererV6

class V6Tests(unittest.TestCase):
 def test_group_inference(self):
  raw={'slug':'x','name':'x','category':'classification','target':'target','metric':'roc_auc','direction':'max','validation':'group-aware','submission_columns':['target'],'risk_flags':['entity_leakage'],'source_url':'x'}
  df=pd.DataFrame({'driver_id':np.repeat(np.arange(20),5),'record_id':np.arange(100),'x':np.random.default_rng(1).normal(size=100),'target':np.tile([0,1],50)})
  s=CompetitionSpecBuilderV6().build(raw); p=CompetitionPlannerV6().inspect(s,df)
  self.assertEqual(p['inferred_validation'],'group-aware'); self.assertIn('record_id',p['data_guard']['drop_feature_columns'])
 def test_rare_event_reliability(self):
  raw={'slug':'x','name':'x','category':'classification','target':'target','metric':'roc_auc','direction':'max','validation':'stratified','submission_columns':['target'],'risk_flags':['imbalance'],'source_url':'x'}
  df=pd.DataFrame({'x':range(1000),'target':[0]*995+[1]*5}); s=CompetitionSpecBuilderV6().build(raw); p=CompetitionPlannerV6().inspect(s,df)
  self.assertEqual(p['rare_event_reliability'],'VERY_LOW')
 def test_complex_metric_review(self):
  raw={'slug':'m5','name':'m5','category':'timeseries','target':'sales','metric':'wrmsse','direction':'min','validation':'rolling-origin','submission_columns':['F1'],'risk_flags':['hierarchy'],'source_url':'x'}
  s=CompetitionSpecBuilderV6().build(raw); p=CompetitionPlannerV6().inspect(s,None)
  self.assertTrue(p['unknowns']); self.assertEqual(p['metric_runtime'],'SPEC_KNOWN_RUNTIME_APPROX')
 def test_renderer(self):
  r=HumanFriendlyRendererV6().review_text(['데이터 확인했음.','결과 비교했음.'],['# 누수를 막기 위해 Train만 fit'],['H-'])
  self.assertEqual(r.score,100)
 def test_system_prepare(self):
  raw={'slug':'h','name':'h','category':'regression','target':'y','metric':'rmse','direction':'min','validation':'kfold','submission_columns':['y'],'risk_flags':[],'source_url':'x'}
  out=V6System(ROOT).prepare_competition(raw,pd.DataFrame({'x':[1,2,3,4,5,6,7,8,9,10],'y':range(10)})); self.assertIn('competition_plan',out); self.assertTrue(out['argument_ledger']['nodes'])
if __name__=='__main__': unittest.main()
