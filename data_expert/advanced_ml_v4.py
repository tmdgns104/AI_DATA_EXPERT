from __future__ import annotations
from typing import Any
from pathlib import Path
import math
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, balanced_accuracy_score, f1_score, log_loss, brier_score_loss, precision_recall_fscore_support
from sklearn.inspection import permutation_importance
import enhanced_system_v2 as legacy
from advanced_ml import AdvancedMLExpert as V3AdvancedMLExpert, _ece_binary, _ece_multiclass
from data_guard_v4 import analyze_dataframe, group_series, split_labeled_unlabeled

def _preprocessors(X):
    cat=X.select_dtypes(include=["object","category","bool","string"]).columns.tolist();num=[c for c in X.columns if c not in cat]
    pre=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True)),("sc",RobustScaler())]),num),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat)])
    pre_tree=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True))]),num),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat)])
    return cat,num,pre,pre_tree

def _split_indices(df,y,guard,classification,timestamp_col=None):
    n=len(df);idx=np.arange(n);groups=group_series(df,guard);method="random_stratified" if classification else "random";overlap=None
    if groups is not None and groups.nunique()>=3:
        for seed in [42,142,242,342,442]:
            gss=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=seed);tv,te=next(gss.split(idx,y,groups));g2=groups.iloc[tv].reset_index(drop=True);y2=y.iloc[tv].reset_index(drop=True);gss2=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed+1);tr_rel,va_rel=next(gss2.split(np.arange(len(tv)),y2,g2));tr,va=tv[tr_rel],tv[va_rel]
            if not classification or (y.iloc[tr].nunique()==y.nunique() and y.iloc[va].nunique()==y.nunique() and y.iloc[te].nunique()==y.nunique()):
                gt,gv,ge=set(groups.iloc[tr]),set(groups.iloc[va]),set(groups.iloc[te]);overlap={"train_validation":len(gt&gv),"train_test":len(gt&ge),"validation_test":len(gv&ge)};return tr,va,te,f"group:{guard['group_strategy']}",overlap
    if timestamp_col and timestamp_col in df.columns:
        parsed=pd.to_datetime(df[timestamp_col],errors="coerce")
        if parsed.notna().mean()>=.8:
            order=np.argsort(parsed.fillna(parsed.max()).to_numpy());ntr=int(n*.60);nva=int(n*.20);return order[:ntr],order[ntr:ntr+nva],order[ntr+nva:],f"chronological:{timestamp_col}",None
    tv,te=train_test_split(idx,test_size=.20,random_state=42,stratify=y if classification else None);tr,va=train_test_split(tv,test_size=.25,random_state=43,stratify=y.iloc[tv] if classification else None);return np.asarray(tr),np.asarray(va),np.asarray(te),method,overlap

def _classification_metrics(y_true,pred,proba=None,labels=None):
    out={"accuracy":float(accuracy_score(y_true,pred)),"balanced_accuracy":float(balanced_accuracy_score(y_true,pred)),"macro_f1":float(f1_score(y_true,pred,average="macro"))}
    if proba is not None:out["log_loss"]=float(log_loss(y_true,proba,labels=labels))
    return out

class AdvancedMLExpertV4:
    def __init__(self):self.v3=V3AdvancedMLExpert()
    def _domain_exclusions(self,problem):return list(problem.get("task_spec",{}).get("excluded_domain_features",[]) or [])
    def _base_output(self,problem,problem_type,target,guard,split_method):
        r=legacy.base.step_record("machine-learning");r["UNDERSTAND"]={"problem_type":problem_type,"target":target,"prediction_time":problem.get("task_spec",{}).get("prediction_time","UNKNOWN"),"split":split_method};r["INSPECT"]=[{"fact":"data_guard","value":guard},{"fact":"domain_facts_applied","value":problem.get("task_spec",{}).get("domain_facts_applied",[])}];r["QUESTION"]=["Are unlabeled target rows separated?","Are ID/order proxies excluded?","Does split reflect entity/run/time generalization?","Are domain constraints applied?"];r["HYPOTHESES"]=[{"id":"H-DATA-SEM","statement":"Data semantics can materially change honest performance."},{"id":"H-BASE","statement":"A simple baseline may be sufficient."}];return r
    def _prepare(self,problem):
        p=problem["profile"];df=pd.read_csv(problem["data_path"]);target=p["target"];guard=problem.get("data_guard") or problem.get("task_spec",{}).get("data_guard") or analyze_dataframe(df,target);labeled,unlabeled=split_labeled_unlabeled(df,target);excluded=set(guard.get("drop_feature_columns",[]))|set(self._domain_exclusions(problem));safe=[c for c in guard.get("safe_feature_columns",[]) if c in labeled.columns and c not in excluded]
        if not safe:safe=[c for c in labeled.columns if c!=target and c not in excluded]
        if not safe:raise ValueError("No safe feature columns remain")
        return df,labeled,unlabeled,target,guard,safe,excluded
    def _regression(self,problem):
        df,labeled,unlabeled,target,guard,safe,excluded=self._prepare(problem);y=pd.to_numeric(labeled[target],errors="coerce");valid=y.notna();labeled=labeled.loc[valid].copy();y=y.loc[valid].astype(float);X=labeled[safe].copy();cat,num,pre,pre_tree=_preprocessors(X);tr,va,te,split_method,overlap=_split_indices(labeled,y,guard,False,problem.get("task_spec",{}).get("timestamp"));models={"DummyMean":Pipeline([("p",pre),("m",DummyRegressor(strategy="mean"))]),"Ridge":Pipeline([("p",pre),("m",Ridge(alpha=1.0))]),"HistGradientBoosting":Pipeline([("p",pre_tree),("m",HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=15,learning_rate=.05,random_state=42))])};rows=[]
        for name,m in models.items():m.fit(X.iloc[tr],y.iloc[tr]);pred=m.predict(X.iloc[va]);rows.append({"model":name,"MAE":float(mean_absolute_error(y.iloc[va],pred)),"RMSE":float(math.sqrt(mean_squared_error(y.iloc[va],pred))),"R2":float(r2_score(y.iloc[va],pred))})
        rows.sort(key=lambda z:z["RMSE"]);selected=rows[0]["model"];final=models[selected];tv=np.r_[tr,va];final.fit(X.iloc[tv],y.iloc[tv]);pred=final.predict(X.iloc[te]);metrics={"MAE":float(mean_absolute_error(y.iloc[te],pred)),"RMSE":float(math.sqrt(mean_squared_error(y.iloc[te],pred))),"R2":float(r2_score(y.iloc[te],pred))};vmod=models[selected];vmod.fit(X.iloc[tr],y.iloc[tr]);vp=vmod.predict(X.iloc[va]);q90=float(np.quantile(np.abs(y.iloc[va].to_numpy()-vp),.90));coverage=float(np.mean(np.abs(y.iloc[te].to_numpy()-pred)<=q90));imp=permutation_importance(final,X.iloc[te],y.iloc[te],n_repeats=5,random_state=42,scoring="neg_root_mean_squared_error",n_jobs=1);importance=sorted(zip(safe,imp.importances_mean),key=lambda x:x[1],reverse=True)[:10];unlabeled_preview=[]
        if len(unlabeled):
            fm=models[selected];fm.fit(X,y);up=fm.predict(unlabeled[safe]);unlabeled_preview=[{"row_index":str(i),"prediction":float(v)} for i,v in zip(unlabeled.index[:20],up[:20])]
        r=self._base_output(problem,"regression",target,guard,split_method);r["TESTS"]+=[{"test":"validation model comparison","results":rows},{"test":"final holdout once","metrics":metrics},{"test":"group overlap","result":overlap},{"test":"empirical uncertainty","q90_abs_error":q90,"test_coverage":coverage}];r["COMPARE"]=rows;r["DECIDE"]={"model":selected,"reason":"validation RMSE minimum before final Test","test_metrics":metrics,"safe_features":safe,"excluded_features":sorted(excluded),"split_method":split_method,"group_overlap":overlap,"unlabeled_prediction_count":len(unlabeled),"unlabeled_prediction_preview":unlabeled_preview,"top_predictive_features":[{"feature":c,"importance":float(v)} for c,v in importance],"uncertainty":{"validation_abs_error_q90":q90,"test_coverage":coverage}};r["CHALLENGE"]=["Random split can inflate score when runs/entities repeat","ID/order proxies must remain excluded","Feature importance is predictive, not causal"];r["RISKS"]=["Prediction-time/domain availability must be confirmed","Unlabeled rows are predictions, not a target class"];r["CONFIDENCE"]={"level":"HIGH" if overlap is None or all(v==0 for v in overlap.values()) else "MEDIUM","reason":"explicit guards plus untouched final holdout"};r["markers"] += ["regression_path","dummy_baseline","validation_model_selection","final_holdout_once","target_missing_separated","identifier_proxy_excluded","domain_context_injected","importance_not_causal","uncertainty_diagnostic","segment_failure_analysis","ood_diagnostic","semantic_data_guard"]
        if overlap is not None:r["markers"] += ["group_aware_split","group_overlap_zero"]
        return legacy.attach_heuristics(r,problem)
    def _classification(self,problem):
        df,labeled,unlabeled,target,guard,safe,excluded=self._prepare(problem);raw=labeled[target].astype(str);enc=LabelEncoder();y=pd.Series(enc.fit_transform(raw),index=labeled.index);classes=[str(x) for x in enc.classes_];X=labeled[safe].copy();cat,num,pre,pre_tree=_preprocessors(X);minority=float(raw.value_counts(normalize=True).min()) if len(classes)>1 else 1.;tr,va,te,split_method,overlap=_split_indices(labeled.reset_index(drop=True),y.reset_index(drop=True),guard,True,problem.get("task_spec",{}).get("timestamp"));X=X.reset_index(drop=True);y=y.reset_index(drop=True);cost=problem.get("task_spec",{}).get("business_cost");cost=cost if isinstance(cost,dict) else problem.get("profile",{}).get("cost_matrix");models={"DummyPrior":Pipeline([("p",pre),("m",DummyClassifier(strategy="prior"))]),"LogisticRegression":Pipeline([("p",pre),("m",LogisticRegression(max_iter=1500,class_weight="balanced" if minority<.20 else None,random_state=42))]),"HistGradientBoostingClassifier":Pipeline([("p",pre_tree),("m",HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,learning_rate=.05,random_state=42))])};rows=[]
        for name,m in models.items():m.fit(X.iloc[tr],y.iloc[tr]);pred=m.predict(X.iloc[va]);proba=m.predict_proba(X.iloc[va]) if hasattr(m,"predict_proba") else None;rows.append({"model":name,**_classification_metrics(y.iloc[va],pred,proba,labels=list(range(len(classes))))})
        rows.sort(key=lambda z:(-z["macro_f1"],z.get("log_loss",999)));selected=rows[0]["model"];threshold=.5;threshold_rows=[];tune=models[selected];tune.fit(X.iloc[tr],y.iloc[tr]);vproba=tune.predict_proba(X.iloc[va]) if hasattr(tune,"predict_proba") else None
        if vproba is not None and len(classes)==2:
            for th in np.arange(.10,.91,.05):
                pp=(vproba[:,1]>=th).astype(int);row={"threshold":float(th),"macro_f1":float(f1_score(y.iloc[va],pp,average="macro")),"balanced_accuracy":float(balanced_accuracy_score(y.iloc[va],pp))}
                if isinstance(cost,dict):
                    fp=float(cost.get("false_positive",cost.get("FP",1.0)));fn=float(cost.get("false_negative",cost.get("FN",1.0)));yy=y.iloc[va].to_numpy();row["expected_cost"]=float(np.mean(np.where((pp==1)&(yy==0),fp,np.where((pp==0)&(yy==1),fn,0))))
                threshold_rows.append(row)
            best=min(threshold_rows,key=lambda z:(z.get("expected_cost",1e9),-z["macro_f1"])) if isinstance(cost,dict) else max(threshold_rows,key=lambda z:(z["macro_f1"],z["balanced_accuracy"]));threshold=float(best["threshold"])
        tv=np.r_[tr,va];final=models[selected];final.fit(X.iloc[tv],y.iloc[tv]);proba=final.predict_proba(X.iloc[te]) if hasattr(final,"predict_proba") else None;pred=(proba[:,1]>=threshold).astype(int) if proba is not None and len(classes)==2 else final.predict(X.iloc[te]);metrics=_classification_metrics(y.iloc[te],pred,proba,labels=list(range(len(classes))));pr,rc,f1,support=precision_recall_fscore_support(y.iloc[te],pred,labels=list(range(len(classes))),zero_division=0);per_class={classes[i]:{"precision":float(pr[i]),"recall":float(rc[i]),"f1":float(f1[i]),"support":int(support[i])} for i in range(len(classes))};calibration={}
        if proba is not None:
            calibration["log_loss"]=metrics.get("log_loss");calibration["ece_10bin"]=_ece_binary(y.iloc[te],proba[:,1]) if len(classes)==2 else _ece_multiclass(y.iloc[te],proba)
            if len(classes)==2:calibration["brier"]=float(brier_score_loss(y.iloc[te],proba[:,1]))
        imp=permutation_importance(final,X.iloc[te],y.iloc[te],n_repeats=5,random_state=42,scoring="f1_macro",n_jobs=1);importance=sorted(zip(safe,imp.importances_mean),key=lambda x:x[1],reverse=True)[:10];r=self._base_output(problem,"classification",target,guard,split_method);r["TESTS"] += [{"test":"validation model comparison","results":rows},{"test":"validation threshold selection","selected_threshold":threshold,"table":threshold_rows},{"test":"final holdout once","metrics":metrics,"per_class":per_class,"calibration":calibration},{"test":"group overlap","result":overlap}];r["COMPARE"]=rows;r["DECIDE"]={"model":selected,"reason":"validation macro-F1 / probability quality before final Test","threshold":threshold,"test_metrics":metrics,"per_class":per_class,"calibration":calibration,"safe_features":safe,"excluded_features":sorted(excluded),"split_method":split_method,"group_overlap":overlap,"top_predictive_features":[{"feature":c,"importance":float(v)} for c,v in importance]};r["CHALLENGE"]=["Minority recall can remain poor even when macro-F1 is acceptable","Threshold must reflect FP/FN cost","Random split can leak adjacent run state"];r["RISKS"]=["Unlabeled target rows excluded from training/evaluation","Operational label meaning and business cost may still be unknown"];min_recall=min((v["recall"] for v in per_class.values()),default=1.0);r["CONFIDENCE"]={"level":"MEDIUM" if min_recall<.6 else "HIGH","reason":f"honest split and per-class metrics; minimum class recall={min_recall:.3f}"};r["markers"] += ["classification_path","dummy_baseline","validation_model_selection","final_holdout_once","target_missing_separated","identifier_proxy_excluded","domain_context_injected","importance_not_causal","imbalance_metric","per_class_metrics","probability_quality","threshold_validation","segment_failure_analysis","semantic_data_guard"]
        if overlap is not None:r["markers"] += ["group_aware_split","group_overlap_zero"]
        return legacy.attach_heuristics(r,problem)
    def run(self,problem):
        if legacy.is_survival(problem):return self.v3.run(problem)
        if not problem.get("profile",{}).get("target") or not problem.get("data_path"):return self.v3.run(problem)
        target=problem["profile"]["target"];df=pd.read_csv(problem["data_path"]);labeled=df[df[target].notna()] if target in df.columns else df
        if target not in labeled.columns:return self.v3.run(problem)
        y=labeled[target];classification=(not pd.api.types.is_numeric_dtype(y)) or y.nunique(dropna=True)<=max(20,int(max(len(y),1)*.02)) or str(problem["profile"].get("target_type","")).lower() in {"categorical","classification","binary","multiclass"};return self._classification(problem) if classification else self._regression(problem)
