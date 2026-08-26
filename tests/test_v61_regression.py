import sys,unittest,tempfile
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'data_expert'))
from enhanced_system import EnhancedSystem
class V61Regression(unittest.TestCase):
 def test_negated_forecast_stays_descriptive(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'x.csv'; pd.DataFrame({'timestamp':pd.date_range('2026-01-01',periods=100,freq='h'),'y':np.arange(100)}).to_csv(p,index=False)
   r=EnhancedSystem().run({'task':'historical time-series analysis only; do not forecast','data_path':str(p),'profile':{'modality':'time-series'}})
   self.assertEqual(r['task_spec']['problem_type'],'descriptive_time_series'); self.assertNotEqual(r['verification']['status'],'FAIL'); self.assertNotIn('time-series',r['routing']['execution_order'])
 def test_explicit_horizon_is_not_unknown(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'x.csv'; n=180; pd.DataFrame({'timestamp':pd.date_range('2026-01-01',periods=n,freq='h'),'y':10+np.sin(np.arange(n)/10)}).to_csv(p,index=False)
   r=EnhancedSystem().run({'task':'forecast the next 24 hours','data_path':str(p),'profile':{'modality':'time-series','target':'y','horizon':'24h','timestamp_col':'timestamp'}})
   self.assertNotIn('explicit forecast horizon',r['task_spec']['unknowns']); self.assertEqual(r['task_spec']['prediction_time'],'next 24h'); self.assertEqual(r['verification']['status'],'PASS')
if __name__=='__main__': unittest.main()
