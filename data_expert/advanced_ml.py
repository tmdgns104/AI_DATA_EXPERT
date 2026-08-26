from __future__ import annotations

from typing import Any
import math
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, balanced_accuracy_score, f1_score, log_loss,
    brier_score_loss, precision_recall_fscore_support,
)

import enhanced_system_v2 as legacy


def _ece_binary(y, p, bins=10):
    y=np.asarray(y).astype(int); p=np.asarray(p,dtype=float); edges=np.linspace(0,1,bins+1); ids=np.clip(np.digitize(p,edges)-1,0,bins-1); ece=0.0
    for b in range(bins):
        m=ids==b
        if m.any(): ece += float(m.mean())*abs(float(y[m].mean())-float(p[m].mean()))
    return float(ece)


def _ece_multiclass(y, proba, bins=10):
    proba=np.asarray(proba); conf=proba.max(axis=1); pred=proba.argmax(axis=1); correct=(pred==np.asarray(y)).astype(float); edges=np.linspace(0,1,bins+1); ids=np.clip(np.digitize(conf,edges)-1,0,bins-1); ece=0.0
    for b in range(bins):
        m=ids==b
        if m.any(): ece += float(m.mean())*abs(float(correct[m].mean())-float(conf[m].mean()))
    return float(ece)


class AdvancedMLExpert:
    def __init__(self):
        self.legacy=legacy.EnhancedML()

    def _preprocessors(self,X):
        cat=X.select_dtypes(include=["object","category","bool","string"]).columns.tolist(); num=[c for c in X.columns if c not in cat]
        pre=ColumnTransformer([
            ("num",Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True)),("sc",RobustScaler())]),num),
            ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat),
        ])
        pre_tree=ColumnTransformer([
            ("num",Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True))]),num),
            ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat),
        ])
        return cat,num,pre,pre_tree

    def _enhance_regression(self,problem,out):
        p=problem["profile"]; df=pd.read_csv(problem["data_path"]); target=p["target"]; X=df.drop(columns=[target]); y=df[target].astype(float); cat,num,pre,pre_tree=self._preprocessors(X)
        model_name=out.get("DECIDE",{}).get("model","HistGradientBoosting")
        models={
            "Dummy":Pipeline([("p",pre),("m",DummyRegressor())]),
            "Ridge":Pipeline([("p",pre),("m",Ridge(alpha=1.0))]),
            "HistGradientBoosting":Pipeline([("p",pre_tree),("m",HistGradientBoostingRegressor(max_iter=180,max_leaf_nodes=15,learning_rate=.05,random_state=42))]),
        }
        model=models.get(model_name,models["HistGradientBoosting"])
        Xtv,Xte,ytv,yte=train_test_split(X,y,test_size=.2,random_state=142)
        Xtr,Xv,ytr,yv=train_test_split(Xtv,ytv,test_size=.25,random_state=142)
        model.fit(Xtr,ytr); vp=model.predict(Xv); q90=float(np.quantile(np.abs(yv.to_numpy()-vp),.90))
        model.fit(Xtv,ytv); pred=model.predict(Xte); abs_err=np.abs(yte.to_numpy()-pred)
        coverage=float(np.mean(abs_err<=q90))
        err=Xte.copy(); err["_abs_error"]=abs_err; segment=[]
        group_candidates=[]
        if problem.get("task_spec",{}).get("group_id") in X.columns: group_candidates.append(problem["task_spec"]["group_id"])
        group_candidates += [c for c in cat if c not in group_candidates]
        for c in group_candidates[:4]:
            g=err.groupby(c,dropna=False)["_abs_error"].agg(["size","mean"]).reset_index(); g=g[g["size"]>=5].sort_values("mean",ascending=False)
            if len(g): segment.append({"feature":c,"worst":g.head(5).to_dict(orient="records")})
        ood=None
        if num:
            tr=Xtr[num].copy(); te=Xte[num].copy(); med=tr.median(); tr=tr.fillna(med); te=te.fillna(med); mu=tr.mean(); sd=tr.std().replace(0,1); dist=np.sqrt((((te-mu)/sd)**2).mean(axis=1)); threshold=float(np.quantile(np.sqrt((((tr-mu)/sd)**2).mean(axis=1)),.95)); mask=dist>threshold
            if mask.any() and (~mask).any(): ood={"threshold":threshold,"ood_rate":float(mask.mean()),"ood_mae":float(abs_err[mask].mean()),"in_distribution_mae":float(abs_err[~mask].mean())}
        advanced={"empirical_interval_abs_error_q90":q90,"test_interval_coverage":coverage,"worst_abs_error_q95":float(np.quantile(abs_err,.95)),"segment_error":segment,"ood_diagnostic":ood}
        out["TESTS"].append({"test":"V3 uncertainty/segment/OOD diagnostics","result":advanced})
        out["DECIDE"]["advanced_diagnostics"]=advanced
        out["CHALLENGE"].append("Overall RMSE can hide segment/OOD failure; inspect advanced diagnostics before deployment")
        out["markers"] += ["uncertainty_diagnostic","segment_failure_analysis","ood_diagnostic","validation_interval","test_isolation_v3"]
        return out

    def _enhance_classification(self,problem,out):
        p=problem["profile"]; df=pd.read_csv(problem["data_path"]); target=p["target"]; X=df.drop(columns=[target]); raw=df[target].astype(str); enc=LabelEncoder(); y=pd.Series(enc.fit_transform(raw),index=raw.index); classes=[str(x) for x in enc.classes_]; cat,num,pre,pre_tree=self._preprocessors(X)
        model_name=out.get("DECIDE",{}).get("model","LogisticRegression")
        minority=float(raw.value_counts(normalize=True).min()) if len(classes)>1 else 1.0
        models={
            "DummyPrior":Pipeline([("p",pre),("m",DummyClassifier(strategy="prior"))]),
            "LogisticRegression":Pipeline([("p",pre),("m",LogisticRegression(max_iter=1200,class_weight="balanced" if minority<.20 else None,random_state=42))]),
            "HistGradientBoostingClassifier":Pipeline([("p",pre_tree),("m",HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,learning_rate=.05,random_state=42))]),
        }
        model=models.get(model_name,models["LogisticRegression"])
        Xtv,Xte,ytv,yte=train_test_split(X,y,test_size=.2,random_state=142,stratify=y)
        Xtr,Xv,ytr,yv=train_test_split(Xtv,ytv,test_size=.25,random_state=142,stratify=ytv)
        model.fit(Xtr,ytr); vproba=model.predict_proba(Xv) if hasattr(model,"predict_proba") else None; threshold=.5; threshold_rows=[]
        cost=p.get("business_cost") or p.get("cost_matrix")
        if vproba is not None and len(classes)==2:
            for th in np.arange(.10,.91,.05):
                pred=(vproba[:,1]>=th).astype(int); f1=float(f1_score(yv,pred,average="macro")); bal=float(balanced_accuracy_score(yv,pred)); row={"threshold":float(th),"macro_f1":f1,"balanced_accuracy":bal}
                if isinstance(cost,dict):
                    fp=float(cost.get("false_positive",cost.get("FP",1.0))); fn=float(cost.get("false_negative",cost.get("FN",1.0))); row["expected_cost"] = float(np.mean(np.where((pred==1)&(yv.to_numpy()==0),fp,np.where((pred==0)&(yv.to_numpy()==1),fn,0.0))))
                threshold_rows.append(row)
            if threshold_rows:
                if isinstance(cost,dict): best=min(threshold_rows,key=lambda r:(r["expected_cost"],-r["macro_f1"]))
                else: best=max(threshold_rows,key=lambda r:(r["macro_f1"],r["balanced_accuracy"]))
                threshold=float(best["threshold"])
        model.fit(Xtv,ytv); proba=model.predict_proba(Xte) if hasattr(model,"predict_proba") else None
        if proba is not None and len(classes)==2: pred=(proba[:,1]>=threshold).astype(int)
        else: pred=model.predict(Xte)
        pr,rc,f1,support=precision_recall_fscore_support(yte,pred,labels=list(range(len(classes))),zero_division=0)
        per_class={classes[i]:{"precision":float(pr[i]),"recall":float(rc[i]),"f1":float(f1[i]),"support":int(support[i])} for i in range(len(classes))}
        calibration={}
        if proba is not None:
            calibration["log_loss"]=float(log_loss(yte,proba,labels=list(range(len(classes)))))
            if len(classes)==2:
                calibration["brier"]=float(brier_score_loss(yte,proba[:,1])); calibration["ece_10bin"]=_ece_binary(yte,proba[:,1])
            else: calibration["ece_10bin"]=_ece_multiclass(yte,proba)
        fail=Xte.copy(); fail["_correct"]=pred==yte.to_numpy(); segment=[]
        group_candidates=[]
        if problem.get("task_spec",{}).get("group_id") in X.columns: group_candidates.append(problem["task_spec"]["group_id"])
        group_candidates += [c for c in cat if c not in group_candidates]
        for c in group_candidates[:4]:
            g=fail.groupby(c,dropna=False)["_correct"].agg(["size","mean"]).reset_index(); g=g[g["size"]>=5].sort_values("mean")
            if len(g): segment.append({"feature":c,"worst":g.head(5).to_dict(orient="records")})
        advanced={"threshold":threshold,"threshold_search":threshold_rows[:20],"calibration":calibration,"per_class":per_class,"segment_accuracy":segment}
        out["TESTS"].append({"test":"V3 threshold/calibration/per-class/segment diagnostics","result":advanced})
        out["DECIDE"]["advanced_diagnostics"]=advanced
        out["CHALLENGE"].append("Threshold 0.5 is not sacred; business cost and calibration can change the operational optimum")
        out["markers"] += ["threshold_validation","probability_quality","per_class_metrics","segment_failure_analysis","test_isolation_v3"]
        return out

    def run(self,problem:dict[str,Any]):
        out=self.legacy.run(problem)
        if legacy.is_survival(problem):return out
        if not problem.get("profile",{}).get("target") or not problem.get("data_path"):return out
        if legacy.infer_supervised_type(problem)=="classification":return self._enhance_classification(problem,out)
        return self._enhance_regression(problem,out)
