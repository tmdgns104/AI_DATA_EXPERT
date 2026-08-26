import sys, unittest
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'data_expert'))
from v6_5_system import V65System
from competition_spec_v6 import CompetitionSpec
from output_renderer_v65 import HumanFriendlyRendererV65

class V65BlindspotTests(unittest.TestCase):
    def setUp(self): self.s=V65System(ROOT)
    def spec(self,validation='group-aware',cat='classification'):
        return CompetitionSpec(slug='x',name='x',category=cat,target='target',metric='roc_auc' if cat=='classification' else 'rmse',direction='maximize' if cat=='classification' else 'minimize',validation=validation,submission_columns=('id','target'),risk_flags=tuple(['entity_leakage'] if validation=='group-aware' else []))
    def test_unseen_group_aliases(self):
        for c in ['crew_ref','family_bundle','merchant_cohort','case_owner','voyage_party']:
            df=pd.DataFrame({'x':np.arange(120)%7,c:np.repeat(np.arange(20),6),'target':np.arange(120)%2})
            p=self.s.competition_planner.inspect(self.spec(),df)
            self.assertIn(c,p['group_candidates']); self.assertEqual(p['inferred_validation'],'group-aware')
    def test_cadence_break(self):
        ts=pd.date_range('2026-01-01',periods=120,freq='h').to_series(index=range(120)); ts.loc[60:]+=pd.Timedelta(hours=5)
        df=pd.DataFrame({'event_time':ts.to_numpy(),'x':np.arange(120),'target':np.arange(120,dtype=float)})
        p=self.s.competition_planner.inspect(self.spec('time-aware','timeseries'),df)
        self.assertTrue(p['time_integrity']['cadence_break_detected']); self.assertEqual(p['time_integrity']['status'],'REVIEW')
    def test_direct_target_copy(self):
        y=np.arange(120)%2; df=pd.DataFrame({'x':np.arange(120)%9,'post_event_measure':y,'target':y})
        p=self.s.competition_planner.inspect(self.spec('stratified','classification'),df)
        self.assertIn('post_event_measure',p['data_guard']['drop_feature_columns']); self.assertIn('post_event_measure',p['critical_leakage_columns'])
    def test_affine_target_proxy(self):
        y=np.linspace(10,80,140); df=pd.DataFrame({'x':np.sin(np.arange(140)),'post_score':2*y+7,'target':y})
        p=self.s.competition_planner.inspect(self.spec('kfold','regression'),df)
        self.assertIn('post_score',p['critical_leakage_columns'])
    def test_human_renderer_v65(self):
        r=HumanFriendlyRendererV65()
        c=r.style_contract()
        self.assertTrue(c['observe_before_labeling_problem'])
        self.assertTrue(c['eda_must_support_next_decision'])
        self.assertTrue(c['validation_selects_test_reports'])
        review=r.review_text(['데이터를 먼저 확인해봄.','Validation에서 정하고 Test 결과를 비교했음.'],['# 미래 정보를 막기 위해 Train만 사용'])
        self.assertEqual(review.score,100)

if __name__=='__main__': unittest.main()
