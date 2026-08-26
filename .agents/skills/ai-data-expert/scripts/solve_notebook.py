#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import textwrap
from pathlib import Path

import nbformat
from nbclient import NotebookClient
import pandas as pd

SCRIPT=Path(__file__).resolve(); ROOT=SCRIPT.parents[4]; CORE=ROOT/'data_expert'; sys.path.insert(0,str(CORE))
from enhanced_system import EnhancedSystem

IDENT=re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")


def extract_problem(nb):
    md='\n\n'.join(c.source for c in nb.cells if c.cell_type=='markdown'); target=None
    for line in md.splitlines():
        lower=line.lower()
        if any(k in lower for k in ['target','예측 목표','목표변수','예측할 변수','예측 모델','예측하세요','분류하세요','예측하고']):
            found=IDENT.findall(line)
            if found: target=found[-1]; break
    return md,target


def json_default(o):
    try:return o.item()
    except Exception:return str(o)


def md(text):return nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip())
def code(text):return nbformat.v4.new_code_cell(textwrap.dedent(text).strip())


def common_cells(data_name,target,expert):
    spec=expert.get('task_spec',{}); guard=expert.get('data_guard') or {}; safe=guard.get('safe_feature_columns',[]); drop=guard.get('drop_feature_columns',[]); group=guard.get('group_strategy'); domain=spec.get('excluded_domain_features',[]) or []; safe=[c for c in safe if c not in domain]
    return [
        md(f'''## 풀이

`{target}`를 목표 변수로 분석함. V4에서는 **타깃 결측 행을 새로운 클래스로 만들지 않고 미라벨 예측 대상으로 분리**하고, `_id`/행순서 같은 식별자·순서 프록시는 모델 입력에서 제외함.

검증 전략: `{spec.get('split_strategy','unknown')}`  
예측 시점: `{spec.get('prediction_time','UNKNOWN')}`  
RAG backend: `{expert.get('domain_context',{}).get('retrieval_backend','none')}`
'''),
        code(f'''import os
os.environ.setdefault("PYTHONUTF8","1")
os.environ.setdefault("PYTHONIOENCODING","utf-8")
os.environ.setdefault("LOKY_MAX_CPU_COUNT","1")
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, balanced_accuracy_score, f1_score, log_loss, classification_report, confusion_matrix, brier_score_loss
from sklearn.inspection import permutation_importance
RANDOM_STATE=42
TARGET={target!r}
DATA_PATH=Path({data_name!r})
SAFE_FEATURES={safe!r}
EXCLUDED_ID_FEATURES={drop!r}
EXCLUDED_DOMAIN_FEATURES={list(domain)!r}
GROUP_STRATEGY={group!r}
df=pd.read_csv(DATA_PATH)
if TARGET not in df.columns: raise KeyError(f"Target {{TARGET!r}} not found")
labeled_df=df[df[TARGET].notna()].copy()
unlabeled_df=df[df[TARGET].isna()].copy()
print("shape:",df.shape,"labeled:",len(labeled_df),"unlabeled:",len(unlabeled_df))
print("excluded ID/order proxy:",EXCLUDED_ID_FEATURES)
print("excluded by domain evidence:",EXCLUDED_DOMAIN_FEATURES)
display(df.head())'''),
        md('### 1. 데이터 품질 / 라벨 상태 / 식별자 Guard'),
        code('''quality=pd.DataFrame({"dtype":df.dtypes.astype(str),"missing":df.isna().sum(),"unique":df.nunique(dropna=True),"unique_ratio":df.nunique(dropna=True)/max(len(df),1)})
display(quality)
print("target missing rows:",int(df[TARGET].isna().sum()))
print("duplicate rows:",int(df.duplicated().sum()))
if len(unlabeled_df): print("미라벨 행은 학습/검증에서 제외하고 마지막에 별도로 예측함")'''),
        md('### 2. 안전 Feature와 Group/Run 생성'),
        code('''candidate_features=[c for c in SAFE_FEATURES if c in labeled_df.columns and c!=TARGET and c not in EXCLUDED_ID_FEATURES and c not in EXCLUDED_DOMAIN_FEATURES]
if not candidate_features: candidate_features=[c for c in labeled_df.columns if c!=TARGET and c not in EXCLUDED_ID_FEATURES and c not in EXCLUDED_DOMAIN_FEATURES]
X=labeled_df[candidate_features].copy(); y=labeled_df[TARGET].copy()

def derive_groups(frame):
    if not GROUP_STRATEGY: return None
    col=GROUP_STRATEGY.get("column")
    if col not in frame.columns:return None
    if GROUP_STRATEGY.get("type")=="column":return frame[col].astype(str).reset_index(drop=True)
    if GROUP_STRATEGY.get("type")=="derived_reset_run":
        x=pd.to_numeric(frame[col],errors="coerce"); span=float(x.max()-x.min()); resets=x.diff()<(-max(span*.15,1.0)); return resets.fillna(False).cumsum().astype(int).reset_index(drop=True)
    return None
GROUPS=derive_groups(labeled_df)
print("features:",candidate_features)
print("group strategy:",GROUP_STRATEGY)
if GROUPS is not None: print("derived/explicit groups:",GROUPS.nunique())'''),
    ]


def split_code(classification: bool):
    strat='y' if classification else 'None'
    return code(f'''idx=np.arange(len(X))
if GROUPS is not None and GROUPS.nunique()>=3:
    chosen=None
    for seed in [42,142,242,342,442]:
        tv,te=next(GroupShuffleSplit(n_splits=1,test_size=.20,random_state=seed).split(idx,y,GROUPS))
        gtv=GROUPS.iloc[tv].reset_index(drop=True); ytv=y.iloc[tv].reset_index(drop=True)
        tr_rel,va_rel=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed+1).split(np.arange(len(tv)),ytv,gtv))
        tr,va=tv[tr_rel],tv[va_rel]
        if {str(classification)} and not (y.iloc[tr].nunique()==y.nunique() and y.iloc[va].nunique()==y.nunique() and y.iloc[te].nunique()==y.nunique()): continue
        chosen=(tr,va,te);break
    if chosen is None: raise RuntimeError("Group split could not preserve required class coverage")
    train_idx,val_idx,test_idx=chosen
    gt,gv,ge=set(GROUPS.iloc[train_idx]),set(GROUPS.iloc[val_idx]),set(GROUPS.iloc[test_idx])
    GROUP_OVERLAP={{"train_validation":len(gt&gv),"train_test":len(gt&ge),"validation_test":len(gv&ge)}}
    SPLIT_METHOD="group-aware"
else:
    train_val_idx,test_idx=train_test_split(idx,test_size=.20,random_state=RANDOM_STATE,stratify={strat})
    train_idx,val_idx=train_test_split(train_val_idx,test_size=.25,random_state=RANDOM_STATE+1,stratify=y.iloc[train_val_idx] if {str(classification)} else None)
    GROUP_OVERLAP=None;SPLIT_METHOD="stratified-random" if {str(classification)} else "random"
print("split:",SPLIT_METHOD,"train/val/test:",len(train_idx),len(val_idx),len(test_idx),"overlap:",GROUP_OVERLAP)''')


def regression_cells(source_name,expert):
    intent=(source_name+' '+str(expert.get('task_spec',{}).get('intent',{}))).lower(); wants_dnn=any(k in intent for k in ['dnn','deep learning','딥러닝','신경망','mlp','neural'])
    model_extra="\nmodels['DNN_MLP']=Pipeline([('p',prep(scale=True)),('m',MLPRegressor(hidden_layer_sizes=(64,32),early_stopping=True,max_iter=450,random_state=RANDOM_STATE))])" if wants_dnn else ''
    import_line="from sklearn.neural_network import MLPRegressor" if wants_dnn else ""
    return [
        md('### 3. Train / Validation / Test 분리'),split_code(False),
        md('### 4. 기준선과 후보 모델 — Validation에서만 선택'),
        code(f'''{import_line}
num=X.select_dtypes(include=np.number).columns.tolist(); cat=[c for c in X.columns if c not in num]
def prep(scale=False):
    ns=[('imputer',SimpleImputer(strategy='median'))]
    if scale:ns.append(('scaler',StandardScaler()))
    return ColumnTransformer([('num',Pipeline(ns),num),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)])
models={{'DummyMean':Pipeline([('p',prep()),('m',DummyRegressor(strategy='mean'))]),'Ridge':Pipeline([('p',prep(scale=True)),('m',Ridge(alpha=1.0))]),'HistGradientBoosting':Pipeline([('p',prep()),('m',HistGradientBoostingRegressor(max_iter=200,learning_rate=.05,random_state=RANDOM_STATE))])}}{model_extra}
rows=[]
for name,m in models.items():
    m.fit(X.iloc[train_idx],y.iloc[train_idx]); pred=m.predict(X.iloc[val_idx]); rows.append({{'Model':name,'MAE':mean_absolute_error(y.iloc[val_idx],pred),'RMSE':np.sqrt(mean_squared_error(y.iloc[val_idx],pred)),'R2':r2_score(y.iloc[val_idx],pred)}})
comparison=pd.DataFrame(rows).sort_values('RMSE').reset_index(drop=True);display(comparison.round(4));selected_name=comparison.loc[0,'Model'];print('validation-selected:',selected_name)'''),
        md('### 5. 선택 모델 최종 Test 1회 평가'),
        code('''selected_model=models[selected_name]; train_val_idx=np.r_[train_idx,val_idx]; selected_model.fit(X.iloc[train_val_idx],y.iloc[train_val_idx]); test_pred=selected_model.predict(X.iloc[test_idx]); test_metrics={'MAE':mean_absolute_error(y.iloc[test_idx],test_pred),'RMSE':np.sqrt(mean_squared_error(y.iloc[test_idx],test_pred)),'R2':r2_score(y.iloc[test_idx],test_pred)};display(pd.DataFrame([test_metrics]).round(4))
# uncertainty width comes from validation residuals only
vm=models[selected_name];vm.fit(X.iloc[train_idx],y.iloc[train_idx]);vp=vm.predict(X.iloc[val_idx]);q90=float(np.quantile(np.abs(y.iloc[val_idx].to_numpy()-vp),.90));coverage=float(np.mean(np.abs(y.iloc[test_idx].to_numpy()-test_pred)<=q90));Empirical90Coverage=coverage; print('validation q90 abs error:',q90,'Empirical90Coverage:',Empirical90Coverage)'''),
        md('### 6. 실패 사례 / 세그먼트 / 중요도'),
        code('''fail=X.iloc[test_idx].copy();fail['actual']=y.iloc[test_idx].to_numpy();fail['prediction']=test_pred;fail['abs_error']=np.abs(fail['actual']-fail['prediction']);display(fail.sort_values('abs_error',ascending=False).head(20))
segment_tables={}
for c in cat[:3]:
    g=fail.groupby(c,dropna=False)['abs_error'].agg(['size','mean']).query('size>=5').sort_values('mean',ascending=False)
    if len(g):segment_tables[c]=g.head(10);print('segment:',c);display(segment_tables[c])
perm=permutation_importance(selected_model,X.iloc[test_idx],y.iloc[test_idx],n_repeats=5,random_state=RANDOM_STATE,scoring='neg_root_mean_squared_error');importance_df=pd.DataFrame({'feature':X.columns,'importance':perm.importances_mean}).sort_values('importance',ascending=False);display(importance_df.head(10).round(4))'''),
        md('### 7. 미라벨 행 예측 및 결론'),
        code('''if len(unlabeled_df):
    final_model=models[selected_name];final_model.fit(X,y);unlabeled_predictions=final_model.predict(unlabeled_df[candidate_features]);unlabeled_result=unlabeled_df.copy();unlabeled_result['predicted_'+TARGET]=unlabeled_predictions;display(unlabeled_result.head(20))
print('selected:',selected_name);print('test:',test_metrics);print('group overlap:',GROUP_OVERLAP);print('Feature importance는 인과관계가 아님. Production 적용 전 예측 시점/비용/공정 규격 확인 필요.')''')
    ]


def classification_cells():
    return [
        md('### 3. Stratified / Group-aware Train / Validation / Test'),split_code(True),
        md('### 4. 기준선과 후보 모델 — Validation macro-F1로 선택'),
        code('''num=X.select_dtypes(include=np.number).columns.tolist();cat=[c for c in X.columns if c not in num]
def prep(scale=False):
    ns=[('imputer',SimpleImputer(strategy='median'))]
    if scale:ns.append(('scaler',StandardScaler()))
    return ColumnTransformer([('num',Pipeline(ns),num),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)])
def metrics(yy,pp,proba=None,labels=None):
    r={'Accuracy':accuracy_score(yy,pp),'BalancedAccuracy':balanced_accuracy_score(yy,pp),'F1_macro':f1_score(yy,pp,average='macro')}
    if proba is not None:r['LogLoss']=log_loss(yy,proba,labels=labels)
    return r
models={'DummyPrior':Pipeline([('p',prep()),('m',DummyClassifier(strategy='prior'))]),'LogisticRegression':Pipeline([('p',prep(scale=True)),('m',LogisticRegression(max_iter=1500,class_weight='balanced',random_state=RANDOM_STATE))]),'HistGradientBoostingClassifier':Pipeline([('p',prep()),('m',HistGradientBoostingClassifier(max_iter=200,learning_rate=.05,random_state=RANDOM_STATE))])}
rows=[]
for name,m in models.items():
    m.fit(X.iloc[train_idx],y.iloc[train_idx]);pred=m.predict(X.iloc[val_idx]);proba=m.predict_proba(X.iloc[val_idx]) if hasattr(m,'predict_proba') else None;rows.append({'Model':name,**metrics(y.iloc[val_idx],pred,proba,labels=m.classes_ if hasattr(m,'classes_') else None)})
comparison=pd.DataFrame(rows).sort_values(['F1_macro','LogLoss'],ascending=[False,True]).reset_index(drop=True);display(comparison.round(4));selected_name=comparison.loc[0,'Model'];print('validation-selected:',selected_name)'''),
        md('### 5. Validation threshold 선택 후 Test 고정 평가'),
        code('''selected_model=models[selected_name];selected_model.fit(X.iloc[train_idx],y.iloc[train_idx]);selected_threshold=.5;threshold_table=None
if y.nunique()==2 and hasattr(selected_model,'predict_proba'):
    cls=selected_model.classes_;vp=selected_model.predict_proba(X.iloc[val_idx])[:,1];vals=[]
    for th in np.arange(.10,.91,.05):
        pp=np.where(vp>=th,cls[1],cls[0]);vals.append({'threshold':float(th),'F1_macro':f1_score(y.iloc[val_idx],pp,average='macro'),'BalancedAccuracy':balanced_accuracy_score(y.iloc[val_idx],pp)})
    threshold_table=pd.DataFrame(vals).sort_values(['F1_macro','BalancedAccuracy'],ascending=False).reset_index(drop=True);selected_threshold=float(threshold_table.loc[0,'threshold']);display(threshold_table.head(10).round(4))
train_val_idx=np.r_[train_idx,val_idx];selected_model=models[selected_name];selected_model.fit(X.iloc[train_val_idx],y.iloc[train_val_idx]);test_proba=selected_model.predict_proba(X.iloc[test_idx]) if hasattr(selected_model,'predict_proba') else None
if test_proba is not None and y.nunique()==2:test_pred=np.where(test_proba[:,1]>=selected_threshold,selected_model.classes_[1],selected_model.classes_[0])
else:test_pred=selected_model.predict(X.iloc[test_idx])
test_metrics=metrics(y.iloc[test_idx],test_pred,test_proba,labels=selected_model.classes_ if hasattr(selected_model,'classes_') else None)
if test_proba is not None and y.nunique()==2:
    pos=selected_model.classes_[1];ybin=(y.iloc[test_idx].to_numpy()==pos).astype(int);prob=test_proba[:,1];test_metrics['Brier']=brier_score_loss(ybin,prob)
    bins=np.linspace(0,1,11);ids=np.clip(np.digitize(prob,bins)-1,0,9);ECE_10bin=0.0
    for b in range(10):
        m=ids==b
        if np.any(m):ECE_10bin+=float(m.mean())*abs(float(ybin[m].mean())-float(prob[m].mean()))
    test_metrics['ECE_10bin']=ECE_10bin
display(pd.DataFrame([{'Model':selected_name,'Threshold':selected_threshold,**test_metrics}]).round(4));print(classification_report(y.iloc[test_idx],test_pred,zero_division=0));display(pd.DataFrame(confusion_matrix(y.iloc[test_idx],test_pred),index=selected_model.classes_,columns=selected_model.classes_))'''),
        md('### 6. 실패 사례 / 세그먼트 / 중요도'),
        code('''fail=X.iloc[test_idx].copy();fail['actual']=y.iloc[test_idx].to_numpy();fail['prediction']=test_pred;fail['correct']=fail['actual']==fail['prediction'];display(fail[~fail['correct']].head(20))
for c in cat[:3]:
    g=fail.groupby(c,dropna=False).agg(n=('correct','size'),accuracy=('correct','mean')).query('n>=5').sort_values('accuracy')
    if len(g):print('segment:',c);display(g.head(10))
perm=permutation_importance(selected_model,X.iloc[test_idx],y.iloc[test_idx],n_repeats=5,random_state=RANDOM_STATE,scoring='f1_macro');importance_df=pd.DataFrame({'feature':X.columns,'importance':perm.importances_mean}).sort_values('importance',ascending=False);display(importance_df.head(10).round(4))'''),
        md('### 7. 미라벨 행 예측 및 운영 판단'),
        code('''if len(unlabeled_df):
    final_model=models[selected_name];final_model.fit(X,y);unlabeled_predictions=final_model.predict(unlabeled_df[candidate_features]);unlabeled_result=unlabeled_df.copy();unlabeled_result['predicted_'+TARGET]=unlabeled_predictions;display(unlabeled_result.head(20))
print('selected:',selected_name);print('test:',test_metrics);print('group overlap:',GROUP_OVERLAP);print('운영 적용은 클래스별 Recall, 오탐/미탐 비용, 실제 예측 시점이 확인되어야 승인할 수 있음.')''')
    ]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--data',required=True);ap.add_argument('--output',required=True);ap.add_argument('--target');ap.add_argument('--context-out');ap.add_argument('--domain-path',action='append',default=[]);ap.add_argument('--timeout',type=int,default=300);args=ap.parse_args()
    inp=Path(args.input).resolve();data=Path(args.data).resolve();out=Path(args.output).resolve();original=nbformat.read(inp,as_version=4);problem_text,target_auto=extract_problem(original);df=pd.read_csv(data);target=args.target or target_auto
    if not target:
        mentioned=[name for name in IDENT.findall(problem_text) if name in df.columns]
        if mentioned:target=mentioned[-1]
    if not target:raise ValueError('Target could not be inferred; pass --target.')
    if target not in df.columns:raise ValueError(f'Target {target!r} not in data')
    labeled_y=df[target].dropna();target_type='continuous' if pd.api.types.is_numeric_dtype(labeled_y) and labeled_y.nunique()>max(20,int(max(len(labeled_y),1)*.02)) else 'categorical';task=(problem_text+'\n'+inp.name).strip();profile={'modality':'tabular','rows':len(df),'columns':df.shape[1],'target':target,'target_type':target_type,'domain_paths':args.domain_path}
    problem={'id':'CODEX-NOTEBOOK-V4','title':inp.stem,'task':task,'data_path':str(data),'profile':profile};expert=EnhancedSystem().run(problem);context_out=Path(args.context_out).resolve() if args.context_out else out.with_suffix('.expert_context.json');context_out.parent.mkdir(parents=True,exist_ok=True);context_out.write_text(json.dumps(expert,ensure_ascii=False,indent=2,default=json_default),encoding='utf-8')
    if expert.get('verification',{}).get('status')=='FAIL':raise RuntimeError(f'Expert verification failed; inspect {context_out}')
    out.parent.mkdir(parents=True,exist_ok=True);local_data=out.parent/data.name
    if local_data.resolve()!=data.resolve():shutil.copy2(data,local_data)
    nb=nbformat.v4.new_notebook();nb.metadata=original.metadata.copy();nb.cells=[c.copy() for c in original.cells];nb.cells+=common_cells(local_data.name,target,expert);nb.cells+=regression_cells(inp.name,expert) if target_type=='continuous' else classification_cells();nbformat.write(nb,out);executed=NotebookClient(nb,timeout=args.timeout,kernel_name='python3',resources={'metadata':{'path':str(out.parent)}}).execute();nbformat.write(executed,out)
    print(json.dumps({'status':'PASS','output':str(out),'expert_context':str(context_out),'route':expert['routing'].get('execution_order'),'target':target,'target_type':target_type,'expert_verification':expert.get('verification',{}).get('status'),'challenger':expert.get('challenger',{}).get('status'),'rag_backend':expert.get('domain_context',{}).get('retrieval_backend')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
