from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CORE=ROOT/'data_expert'
sys.path.insert(0,str(CORE))
from enhanced_system import EnhancedSystem
from data_guard_v4 import analyze_dataframe


class V4ImprovementTests(unittest.TestCase):
    def _diecast_like(self, path: Path, missing: int = 24):
        rng=np.random.default_rng(12);rows=[]
        for run in range(10):
            shift=rng.normal(0,.6)
            for shot in range(1,61):
                temp=100+shift+rng.normal(0,1);pressure=50+rng.normal(0,2);status=1 if (shot<7 or temp>102.2 or rng.random()<.025) else 0
                rows.append({'_id':len(rows)+10000,'Shot':shot,'Velocity_1':rng.normal(2,.2),'Temperature':temp,'Pressure':pressure,'Machine_Status':status})
        df=pd.DataFrame(rows);idx=rng.choice(df.index,missing,replace=False);df.loc[idx,'Machine_Status']=np.nan;df.to_csv(path,index=False);return df

    def test_01_target_missing_id_and_run_guard(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'die.csv';df=self._diecast_like(p)
            g=analyze_dataframe(df,'Machine_Status')
            self.assertEqual(g['target_missing_count'],24);self.assertIn('_id',g['drop_feature_columns']);self.assertEqual(g['group_strategy']['type'],'derived_reset_run');self.assertEqual(g['group_strategy']['n_groups'],10)

    def test_02_honest_group_split_and_unlabeled_prediction(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'die.csv';df=self._diecast_like(p)
            prob={'id':'die','task':'build a model to predict Machine_Status from die casting sensor data','data_path':str(p),'profile':{'modality':'tabular','target':'Machine_Status','target_type':'categorical','rows':len(df),'prediction_time':'before normal-operation decision','business_cost':{'false_positive':1,'false_negative':8}}}
            r=EnhancedSystem().run(prob);self.assertEqual(r['verification']['status'],'PASS');ml=next(o for o in r['expert_outputs'] if o['agent']=='machine-learning');d=ml['DECIDE'];self.assertEqual(d['unlabeled_prediction_count'],24);self.assertIn('_id',d['excluded_features']);self.assertTrue(all(v==0 for v in d['group_overlap'].values()));self.assertIn('group_aware_split',ml['markers'])

    def test_03_hybrid_rag_returns_component_scores_and_text(self):
        p=ROOT/'examples/4_manufacturing_yield.csv';prob={'id':'rag','task':'predict manufacturing yield using processing_time_sec','data_path':str(p),'profile':{'modality':'tabular','target':'yield_percentage','target_type':'continuous','domain':'manufacturing'}}
        r=EnhancedSystem().run(prob);dc=r['domain_context'];self.assertEqual(dc['status'],'FOUND');self.assertTrue(dc['retrieval_backend']);m=dc['matches'][0];self.assertIn('bm25_score',m);self.assertIn('vector_score',m);self.assertTrue(m['text']);self.assertTrue(r['task_spec']['domain_facts_applied']);self.assertIn('processing_time_sec',r['task_spec']['excluded_domain_features'])

    def test_04_domain_fact_target_scope_prevents_cross_task_pollution(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.csv';pd.DataFrame({'sensor':[1,2,3,4,5,6]*20,'Machine_Status':[0,0,0,1,1,1]*20}).to_csv(p,index=False)
            prob={'id':'x','task':'classify Machine_Status in manufacturing','data_path':str(p),'profile':{'modality':'tabular','target':'Machine_Status','target_type':'categorical','domain':'manufacturing','prediction_time':'decision time','business_cost':'review'}}
            r=EnhancedSystem().run(prob);self.assertEqual(r['task_spec'].get('domain_facts_applied'),[]);self.assertFalse(r['task_spec'].get('excluded_domain_features'))

    def test_05_no_training_vision_does_not_false_fail_verifier(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'v.csv';pd.DataFrame({'label':[0,1]*20,'product_id':np.arange(40)}).to_csv(p,index=False)
            r=EnhancedSystem().run({'id':'v','task':'audit CNN image labels only; do not train a model','data_path':str(p),'profile':{'modality':'image','target':'label','rows':40}});self.assertNotIn('deep-learning',r['routing']['execution_order']);self.assertEqual(r['verification']['status'],'PASS')

    def test_06_windows_setup_propagates_failure_and_utf8(self):
        text=(ROOT/'setup_windows.bat').read_text(encoding='utf-8').lower();self.assertGreaterEqual(text.count('if errorlevel 1 goto :fail'),4);self.assertIn('set pythonutf8=1',text);self.assertIn('exit /b 1',text);self.assertLess(text.index('python -m pip install -r requirements.txt'),text.index('echo setup complete.'))

    def test_07_three_class_pixel_cnn_improved(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td);rng=np.random.default_rng(22);n=180;labels=np.arange(n)%3;images=np.zeros((n,1,18,18),dtype=np.float32)
            for i,c in enumerate(labels):
                if c==0:images[i,0,2:7,2:7]=1
                elif c==1:images[i,0,6:12,6:12]=1
                else:images[i,0,11:16,2:8]=1
            images += rng.normal(0,.12,images.shape);npz=td/'img.npz';np.savez(npz,images=images,labels=labels);meta=td/'meta.csv';pd.DataFrame({'label':labels,'product_id':np.arange(n)}).to_csv(meta,index=False)
            r=EnhancedSystem().run({'id':'vision3','task':'train a three class CNN image classifier','data_path':str(meta),'profile':{'modality':'image','target':'label','rows':n,'image_npz':str(npz),'prediction_time':'image capture','business_cost':'class misses costly'}});dl=next(o for o in r['expert_outputs'] if o['agent']=='deep-learning');metrics=dl['DECIDE']['validation_metrics'];self.assertGreaterEqual(metrics['accuracy'],.85);self.assertGreaterEqual(metrics['macro_f1'],.80);self.assertEqual(r['verification']['status'],'PASS')

    def test_08_notebook_solver_missing_target_group_and_semantic_validation(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td);data=td/'die.csv';self._diecast_like(data,missing=12);q=td/'question.ipynb';nb=nbformat.v4.new_notebook();nb.cells=[nbformat.v4.new_markdown_cell('''# Die casting task\nUse the data to predict `Machine_Status`. Explain preprocessing and model selection.''')];nbformat.write(nb,q);out=td/'answer.ipynb';solver=ROOT/'.agents/skills/ai-data-expert/scripts/solve_notebook.py';validator=ROOT/'.agents/skills/ai-data-expert/scripts/validate_notebook.py'
            subprocess.run([sys.executable,str(solver),'--input',str(q),'--data',str(data),'--output',str(out),'--target','Machine_Status'],check=True,capture_output=True,text=True,timeout=300)
            cp=subprocess.run([sys.executable,str(validator),str(out),'--data',str(data),'--target','Machine_Status'],check=True,capture_output=True,text=True,timeout=300);payload=json.loads(cp.stdout.strip().splitlines()[-1]);self.assertEqual(payload['semantic']['status'],'PASS');code='\n'.join(c.source for c in nbformat.read(out,as_version=4).cells if c.cell_type=='code');self.assertIn('labeled_df=df[df[TARGET].notna()]',code);self.assertIn('GroupShuffleSplit',code);self.assertIn("EXCLUDED_ID_FEATURES=['_id']",code)

if __name__=='__main__':unittest.main()
