from __future__ import annotations
import hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np,pandas as pd,nbformat

ROOT=Path(__file__).resolve().parents[1]; SKILL=ROOT/'.agents/skills/ai-data-expert/SKILL.md'; CORE=ROOT/'data_expert'; sys.path.insert(0,str(CORE))
from enhanced_system import EnhancedSystem

class V3SkillTests(unittest.TestCase):
    def _tabular(self,task,target_type='continuous',prediction_time=None,business_cost=None):
        td=tempfile.TemporaryDirectory(); self.addCleanup(td.cleanup); path=Path(td.name)/'data.csv'; rng=np.random.default_rng(11); n=360
        if target_type=='continuous':
            x1=rng.normal(size=n);x2=rng.normal(size=n);df=pd.DataFrame({'x1':x1,'x2':x2,'segment':np.where(np.arange(n)%2==0,'A','B'),'target':2*x1-.7*x2+rng.normal(0,.3,n)});target='target'
        else:
            x1=rng.normal(size=n);x2=rng.normal(size=n);df=pd.DataFrame({'x1':x1,'x2':x2,'segment':np.where(np.arange(n)%2==0,'A','B'),'label':np.where(x1+.3*x2>0,'yes','no')});target='label'
        df.to_csv(path,index=False); profile={'modality':'tabular','target':target,'target_type':target_type,'rows':n}
        if prediction_time:profile['prediction_time']=prediction_time
        if business_cost:profile['business_cost']=business_cost
        return {'id':'case','task':task,'data_path':str(path),'profile':profile}

    def test_01_codex_skill_discoverable(self):
        self.assertTrue(SKILL.exists()); text=SKILL.read_text(encoding='utf-8'); self.assertRegex(text,r'(?m)^name:\s*ai-data-expert\s*$'); self.assertTrue((ROOT/'AGENTS.md').exists())

    def test_02_parent_tacit_provenance_unchanged(self):
        freeze=json.loads((CORE/'PARENT_TACIT_V2_FREEZE.json').read_text(encoding='utf-8')); self.assertEqual(hashlib.sha256((CORE/'tacit_knowledge/HEURISTICS.json').read_bytes()).hexdigest(),freeze['heuristics_sha256']); self.assertEqual(hashlib.sha256((CORE/'tacit_knowledge/SOURCES.json').read_bytes()).hexdigest(),freeze['sources_sha256'])

    def test_03_estimate_wording_routes_ml(self):
        r=EnhancedSystem().run(self._tabular('estimate target from the available sensor readings','continuous','before process completion','MAE under 0.5'))
        self.assertEqual(r['task_spec']['intent']['primary_intent'],'TRAIN_MODEL'); self.assertIn('machine-learning',r['routing']['execution_order']); self.assertEqual(r['verification']['status'],'PASS')

    def test_04_taskspec_is_structured(self):
        r=EnhancedSystem().run(self._tabular('train a regression model','continuous','before shipment','false large errors are costly'))
        s=r['task_spec']; self.assertEqual(s['problem_type'],'regression'); self.assertEqual(s['prediction_time'],'before shipment'); self.assertIn('RMSE',s['primary_metric']); self.assertIn('observation_unit',s); self.assertIn('available_features',s)

    def test_05_string_classification_advanced_diagnostics(self):
        r=EnhancedSystem().run(self._tabular('build a classification model and compare baselines','categorical','decision time','false negative cost high'))
        self.assertEqual(r['verification']['status'],'PASS'); ml=next(o for o in r['expert_outputs'] if o['agent']=='machine-learning'); ms=set(ml['markers']);
        for marker in ['classification_path','probability_quality','threshold_validation','per_class_metrics','segment_failure_analysis','final_holdout_once']:self.assertIn(marker,ms)

    def test_06_regression_uncertainty_segment_ood(self):
        r=EnhancedSystem().run(self._tabular('predict target and compare models','continuous','before outcome','RMSE is primary'))
        self.assertEqual(r['verification']['status'],'PASS'); ml=next(o for o in r['expert_outputs'] if o['agent']=='machine-learning'); ms=set(ml['markers']);
        for marker in ['uncertainty_diagnostic','segment_failure_analysis','ood_diagnostic','final_holdout_once']:self.assertIn(marker,ms)

    def test_07_actual_pytorch_tabular_dl(self):
        r=EnhancedSystem().run(self._tabular('train a deep learning neural network classifier and compare it with a simple model','categorical','decision time','macro F1'))
        self.assertEqual(r['routing']['primary_agent'],'deep-learning'); dl=next(o for o in r['expert_outputs'] if o['agent']=='deep-learning'); self.assertIn('actual_torch_training',dl['markers']); self.assertIn('small_batch_overfit',dl['markers']); self.assertNotEqual(r['verification']['status'],'FAIL')

    def test_08_actual_pixel_cnn(self):
        td=tempfile.TemporaryDirectory(); self.addCleanup(td.cleanup); td=Path(td.name); rng=np.random.default_rng(4); n=100; labels=np.arange(n)%2; images=np.zeros((n,1,12,12),dtype=np.float32)
        for i in range(n): images[i,0,2:5,2:5]=1 if labels[i]==0 else 0; images[i,0,7:10,7:10]=1 if labels[i]==1 else 0
        images+=rng.normal(0,.04,images.shape); npz=td/'img.npz'; np.savez(npz,images=images,labels=labels); meta=td/'meta.csv'; pd.DataFrame({'label':labels,'product_id':np.arange(n)}).to_csv(meta,index=False)
        prob={'id':'img','task':'train a CNN image classifier on the pixel images','data_path':str(meta),'profile':{'modality':'image','target':'label','rows':n,'image_npz':str(npz),'prediction_time':'image capture','business_cost':'false negative costly'}}
        r=EnhancedSystem().run(prob); dl=next(o for o in r['expert_outputs'] if o['agent']=='deep-learning'); self.assertIn('actual_pixel_training',dl['markers']); self.assertEqual(r['verification']['status'],'PASS')

    def test_09_no_train_negation(self):
        td=tempfile.TemporaryDirectory();self.addCleanup(td.cleanup);p=Path(td.name)/'v.csv';pd.DataFrame({'label':[0,1]*20,'product_id':np.arange(40)}).to_csv(p,index=False)
        r=EnhancedSystem().run({'id':'v','task':'audit image labels and duplicates only; do not train a model','data_path':str(p),'profile':{'modality':'image','target':'label','rows':40}}); self.assertNotIn('deep-learning',r['routing']['execution_order']); self.assertEqual(r['verification']['status'],'PASS')

    def test_10_no_forecast_negation(self):
        td=tempfile.TemporaryDirectory();self.addCleanup(td.cleanup);p=Path(td.name)/'t.csv';pd.DataFrame({'timestamp':pd.date_range('2026-01-01',periods=120,freq='h'),'y':np.arange(120)}).to_csv(p,index=False)
        r=EnhancedSystem().run({'id':'t','task':'historical time-series analysis only; do not forecast','data_path':str(p),'profile':{'modality':'time-series','rows':120}}); self.assertNotIn('time-series',r['routing']['execution_order']); self.assertEqual(r['verification']['status'],'PASS')

    def test_11_domain_rag_retrieves_manufacturing_constraint(self):
        prob=self._tabular('predict manufacturing yield using processing_time_sec and power_consumption','continuous','before process completion','RMSE')
        p=Path(prob['data_path']);df=pd.read_csv(p);df['processing_time_sec']=np.arange(len(df));df['power_consumption']=1.0;p2=p.with_name('manufacturing.csv');df.to_csv(p2,index=False);prob['data_path']=str(p2)
        r=EnhancedSystem().run(prob); self.assertEqual(r['domain_context']['status'],'FOUND'); text=' '.join(m['text'] for m in r['domain_context']['matches']); self.assertIn('prediction-time availability',text)

    def test_12_hypothesis_and_experiment_layers(self):
        r=EnhancedSystem().run(self._tabular('predict target','continuous','before outcome','MAE')); self.assertGreaterEqual(len(r['hypotheses']),4); self.assertEqual(r['experiment_evidence']['status'],'PASS'); tests={e['test'] for e in r['experiment_evidence']['evidence']}; self.assertIn('duplicate_rate',tests)

    def test_13_unknown_operational_context_returns_review_not_false_pass(self):
        r=EnhancedSystem().run(self._tabular('train a regression model to predict target')); self.assertEqual(r['verification']['status'],'REVIEW'); codes={i['code'] for i in r['challenger']['issues']}; self.assertIn('PREDICTION_TIME_UNKNOWN',codes)

    def test_14_deployment_requires_prediction_time(self):
        prob=self._tabular('deploy this trained regression model to production','continuous'); prob['profile']['deployment']=True
        r=EnhancedSystem().run(prob); self.assertEqual(r['verification']['status'],'FAIL'); failed={c['name'] for c in r['verification']['checks'] if not c['pass']}; self.assertIn('deployment_prediction_time',failed)

    def test_15_causal_request_gets_causal_guard(self):
        prob=self._tabular('which sensor causes target to improve? train a predictive model too','continuous','before outcome','MAE')
        r=EnhancedSystem().run(prob); self.assertIn('CAUSAL_ANALYSIS',r['task_spec']['intent']['intents']); ml=next(o for o in r['expert_outputs'] if o['agent']=='machine-learning'); self.assertIn('prediction_vs_causality',ml['markers']); self.assertNotEqual(r['verification']['status'],'FAIL')

    def test_16_expert_failure_isolated(self):
        td=tempfile.TemporaryDirectory();self.addCleanup(td.cleanup);p=Path(td.name)/'bad.csv';pd.DataFrame({'x':[1,2,3,4,5]}).to_csv(p,index=False)
        r=EnhancedSystem().run({'id':'bad','task':'estimate absent_target from x','data_path':str(p),'profile':{'modality':'tabular','target':'absent_target','target_type':'continuous','rows':5}}); self.assertEqual(r['verification']['status'],'FAIL'); self.assertTrue(r['expert_errors']); self.assertIn('data-analyst',[o['agent'] for o in r['expert_outputs']])

    def test_17_forecast_routes_and_uses_naive(self):
        td=tempfile.TemporaryDirectory();self.addCleanup(td.cleanup);p=Path(td.name)/'ts.csv';n=360;pd.DataFrame({'timestamp':pd.date_range('2026-01-01',periods=n,freq='h'),'y':10+np.sin(np.arange(n)*2*np.pi/24)}).to_csv(p,index=False)
        r=EnhancedSystem().run({'id':'ts','task':'forecast the next 24 hours and compare seasonal naive','data_path':str(p),'profile':{'modality':'time-series','target':'y','timestamp_col':'timestamp','horizon':'24h','rows':n}}); self.assertIn('time-series',r['routing']['execution_order']); ts=next(o for o in r['expert_outputs'] if o['agent']=='time-series'); self.assertIn('seasonal_naive',ts['markers']); self.assertEqual(r['verification']['status'],'PASS')

    def test_18_survival_censoring(self):
        td=tempfile.TemporaryDirectory();self.addCleanup(td.cleanup);p=Path(td.name)/'s.csv';rng=np.random.default_rng(5);n=180;pd.DataFrame({'days':rng.exponential(30,n)+1,'event':rng.binomial(1,.6,n),'temp':rng.normal(size=n)}).to_csv(p,index=False)
        r=EnhancedSystem().run({'id':'s','task':'analyze censored time to failure','data_path':str(p),'profile':{'modality':'tabular','target':'days','censor_col':'event','rows':n}}); ml=next(o for o in r['expert_outputs'] if o['agent']=='machine-learning'); self.assertIn('survival_censoring',ml['markers']); self.assertEqual(r['verification']['status'],'PASS')

    def test_19_mlops_no_auto_retrain(self):
        td=tempfile.TemporaryDirectory();self.addCleanup(td.cleanup);p=Path(td.name)/'m.csv';rng=np.random.default_rng(6);ref=rng.normal(0,1,100);cur=rng.normal(.6,1,100);a=np.r_[5+ref,5+cur];pred=a+rng.normal(0,.3,200);pd.DataFrame({'period':['reference']*100+['current']*100,'f':np.r_[ref,cur],'actual':a,'prediction':pred}).to_csv(p,index=False)
        r=EnhancedSystem().run({'id':'m','task':'monitor deployed model drift; do not retrain automatically','data_path':str(p),'profile':{'modality':'tabular','monitoring':True,'existing_model':True,'rows':200}}); self.assertEqual(r['routing']['primary_agent'],'mlops'); mlops=next(o for o in r['expert_outputs'] if o['agent']=='mlops'); self.assertIn('no_auto_retrain',mlops['markers']); self.assertEqual(r['verification']['status'],'PASS')

    def test_20_bigdata_architecture(self):
        r=EnhancedSystem().run({'id':'b','task':'design a reliable real-time architecture for 2TB event streams','data_path':None,'profile':{'modality':'big-data','size_gb':2048,'rows':2_000_000_000,'streaming':True,'latency_sla':'3 seconds'}}); self.assertIn('big-data',r['routing']['execution_order']); self.assertEqual(r['verification']['status'],'PASS')

    def test_21_regression_notebook_solver_executes_advanced_analysis(self):
        q=ROOT/'examples/DNN_regression_question.ipynb';data=ROOT/'examples/4_manufacturing_yield.csv';out=ROOT/'outputs/test_v3_regression.ipynb';script=ROOT/'.agents/skills/ai-data-expert/scripts/solve_notebook.py'
        subprocess.run([sys.executable,str(script),'--input',str(q),'--data',str(data),'--output',str(out)],check=True,capture_output=True,text=True,timeout=300);nb=nbformat.read(out,as_version=4);code='\n'.join(c.source for c in nb.cells if c.cell_type=='code');self.assertIn('Empirical90Coverage',code);self.assertIn('segment_tables',code);self.assertIn('DNN_MLP',code);ctx=json.loads(out.with_suffix('.expert_context.json').read_text(encoding='utf-8'));self.assertNotEqual(ctx['verification']['status'],'FAIL')

    def test_22_classification_notebook_solver_threshold_and_calibration(self):
        q=ROOT/'examples/classification_question.ipynb';data=ROOT/'examples/classification_example.csv';out=ROOT/'outputs/test_v3_classification.ipynb';script=ROOT/'.agents/skills/ai-data-expert/scripts/solve_notebook.py'
        subprocess.run([sys.executable,str(script),'--input',str(q),'--data',str(data),'--output',str(out)],check=True,capture_output=True,text=True,timeout=300);nb=nbformat.read(out,as_version=4);code='\n'.join(c.source for c in nb.cells if c.cell_type=='code');self.assertIn('selected_threshold',code);self.assertIn('Brier',code);self.assertIn('ECE_10bin',code);self.assertIn('classification_report',code)

if __name__=='__main__':unittest.main()
