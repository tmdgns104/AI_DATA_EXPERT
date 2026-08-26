from __future__ import annotations
import sys,unittest
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'data_expert'))
from v6_3_system import V63System

class V63Generalization(unittest.TestCase):
 def test_entity_key_generalizes(self):
  n=120; df=pd.DataFrame({'x':np.arange(n),'entity_key':np.repeat(np.arange(20),6),'target':np.arange(n)%2})
  raw={'category':'classification','slug':'x','name':'x','metric':'accuracy','direction':'max','target':'target','validation':'group-aware','submission_columns':['target'],'risk_flags':['entity_leakage'],'source_url':'x'}
  p=V63System(ROOT).prepare_competition(raw,df)['competition_plan']; self.assertIn('entity_key',p['group_candidates']); self.assertEqual(p['inferred_validation'],'group-aware')
 def test_time_reversal_flagged(self):
  n=120; t=pd.date_range('2026-01-01',periods=n,freq='h').to_numpy(); t[[50,51]]=t[[51,50]]
  df=pd.DataFrame({'event_time':t,'x':np.arange(n),'y':np.arange(n)})
  raw={'category':'regression','slug':'x','name':'x','metric':'rmse','direction':'min','target':'y','validation':'time-aware','submission_columns':['y'],'risk_flags':['time_order'],'source_url':'x'}
  p=V63System(ROOT).prepare_competition(raw,df)['competition_plan']; self.assertTrue(p['time_integrity']['non_monotonic_detected'])
 def test_bundled_demo_rejected_for_generic_targets(self):
  for target in ['y','target','loss','sales','label']:
   r=V63System(ROOT).rag.retrieve({'task':'solve external competition','profile':{'target':target,'modality':'tabular'}})
   self.assertEqual(r['status'],'NO_MATCH',target); self.assertEqual(r['facts'],[],target)
if __name__=='__main__':unittest.main()
