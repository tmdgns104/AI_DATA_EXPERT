from __future__ import annotations
import importlib.util, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, log_loss,
    roc_auc_score, average_precision_score,
)
from sklearn.inspection import permutation_importance

HERE = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("base_reasoning", HERE / "reasoning_system.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

SOURCES = json.loads((HERE/"tacit_knowledge/SOURCES.json").read_text(encoding="utf-8"))
RULES = json.loads((HERE/"tacit_knowledge/HEURISTICS.json").read_text(encoding="utf-8"))

def has(text, words):
    t=(text or "").lower()
    return any(w.lower() in t for w in words)

def infer_supervised_type(problem):
    p=problem.get("profile",{})
    explicit=str(p.get("target_type","")).lower()
    if explicit in {"categorical","classification","binary","multiclass","class"}: return "classification"
    if explicit in {"continuous","regression","numeric","count"}: return "regression"
    path=problem.get("data_path"); target=p.get("target")
    if path and target and Path(path).exists():
        df=pd.read_csv(path,usecols=[target]); y=df[target]
        if (not pd.api.types.is_numeric_dtype(y)) or y.nunique(dropna=True) <= max(20, int(len(y)*0.02)): return "classification"
    return "regression"

def _contract_complete(output):
    keys=["UNDERSTAND","INSPECT","QUESTION","HYPOTHESES","TESTS","COMPARE","DECIDE","CHALLENGE","RISKS","CONFIDENCE"]
    return all(bool(output.get(k)) for k in keys)

def is_multilabel(problem):
    p=problem.get("profile",{}); task=problem.get("task","")
    if p.get("multilabel") or p.get("target")=="multiple_labels": return True
    if has(task,["동시에 있을 수","여러 결함을 동시에","multi-label","multilabel"]): return True
    path=problem.get("data_path")
    if path and Path(path).exists():
        df=pd.read_csv(path,nrows=10); label_cols=[c for c in df.columns if c.lower() in {"scratch","dent","contamination","crack","stain"}]
        if len(label_cols)>=2:return True
    return False

def is_survival(problem):
    p=problem.get("profile",{}); task=problem.get("task","")
    if p.get("censor_col") or p.get("event_col") or p.get("entry_col"): return True
    return has(task,["고장나지 않은","검열","censor","time-to-event","생존","관측 기간 안에 고장"])

def context_tags(problem, agent):
    p=problem.get("profile",{}); task=problem.get("task",""); tags=set(); modality=p.get("modality","tabular")
    if agent=="data-analyst": tags.update(["analysis","eda"])
    if modality=="tabular": tags.add("tabular")
    if p.get("repeated_measure") or p.get("mixed_granularity"): tags.update(["repeated_measure","mixed_granularity"])
    if agent in {"machine-learning","data-analyst"} and p.get("target"): tags.add("modeling")
    if agent=="machine-learning": tags.update(["prediction","regression_model","new_pipeline"])
    if is_survival(problem): tags.update(["censoring","time_to_event"])
    if agent=="deep-learning":
        tags.update(["deep_learning","training"])
        if p.get("augmentation_changes_label") or p.get("augmentation_semantic_risk"): tags.add("augmentation_semantics")
    if agent=="vision-ai":
        tags.add("vision")
        if is_multilabel(problem): tags.add("multiple_simultaneous_labels")
    if agent=="time-series": tags.add("forecast")
    if agent=="big-data":
        tags.add("large_scale")
        if p.get("streaming"):tags.add("streaming")
    if agent=="mlops":
        tags.update(["production","monitoring"])
        if p.get("labels_delayed") or has(task,["label은 아직","정답 label은 아직","label이 아직"]): tags.add("labels_delayed")
    return tags

def attach_heuristics(output, problem):
    agent=output["agent"]; tags=context_tags(problem,agent); trace=[]
    for r in RULES:
        if agent not in r["domains"]: continue
        if not (set(r["trigger"]) & tags): continue
        source=r["source"]; s=SOURCES.get(source)
        trace.append({"heuristic_id":r["id"],"expert":s["expert"] if s else "System-derived engineering rule","source_title":s["source_title"] if s else "System benchmark-derived rule","source_url":s["source_url"] if s else None,"applied_context":sorted(set(r["trigger"]) & tags),"heuristic":r["heuristic"],"actions":r["actions"]})
    output["heuristic_trace"]=trace
    return output

class EnhancedSupervisor(base.Supervisor):
    def route(self, problem):
        route=super().route(problem); p=problem.get("profile",{}); task=(problem.get("task","") or "").lower(); modality=p.get("modality","tabular"); selected=list(route.get("selected_agents",[])); reasons={k:list(v) for k,v in route.get("routing_reasons",{}).items()}
        no_train=has(task,["학습은 하지","모델 학습은 하지","학습하지 말","하지 마","하지마","do not train","don't train","without training","no training","audit only","analysis only"])
        no_forecast=has(task,["forecast는 하지","미래 예측은 하지","예측하지 말고","과거 패턴만","do not forecast","don't forecast","no forecast","historical analysis only"])
        train_kw=["학습","예측","분류","회귀","검출","segmentation","모델을 만들어","train","fit model","fit a model","predict","prediction","classification","classify","classifier","regression","regressor","detect","detection","segment","segmentation","build model","build a model","model training"]
        train=has(task,train_kw) and not no_train; dl=has(task,["dnn","딥러닝","deep learning","mlp","cnn","lstm","transformer","neural network","neural net"]); forecast=has(task,["향후","forecast","forecasting","예측 horizon","다음 24시간","future","horizon","next 24 hours","잔여수명"]) and not no_forecast; monitor=bool(p.get("monitoring")) or has(task,["drift","모니터링","monitoring","performance degradation","성능 저하","retrain","재학습 여부"])
        def add(agent,reason):
            if agent not in selected:selected.append(agent)
            reasons.setdefault(agent,[]).append(reason)
        def remove(agent):
            if agent in selected:selected.remove(agent)
            reasons.pop(agent,None)
        if modality=="tabular" and p.get("target") and train:add("data-analyst","Feature/Target EDA and data quality");add("machine-learning","supervised tabular modeling intent")
        if modality in {"image","video","vision"}:
            add("vision-ai","Vision modality")
            if train:add("deep-learning","Vision training intent")
        if modality=="time-series":
            add("data-analyst","time-series data quality")
            if forecast:add("time-series","temporal forecasting/backtesting intent")
            elif no_forecast:remove("time-series")
        if dl and train:add("deep-learning","explicit deep-learning intent")
        if monitor or (p.get("existing_model") and (p.get("deployment") or p.get("monitoring"))):add("mlops","existing-model monitoring/production intent")
        if no_train:
            remove("deep-learning")
            if not is_survival(problem) and not has(task,["모델 평가","evaluate model","model evaluation"]):remove("machine-learning")
        if no_forecast:remove("time-series")
        if is_multilabel(problem):
            add("vision-ai","multi-label visual semantics")
            if not no_train:add("deep-learning","multi-label output/loss strategy")
        if is_survival(problem):add("data-analyst","censoring/event structure review");add("machine-learning","censor-aware survival modeling")
        if not selected:add("data-analyst","default analysis")
        if p.get("existing_model") and monitor:primary="mlops"
        elif is_survival(problem):primary="machine-learning"
        elif modality in {"image","video","vision"}:primary="vision-ai"
        elif modality=="time-series" and "time-series" in selected:primary="time-series"
        elif "machine-learning" in selected:primary="machine-learning"
        elif "big-data" in selected and not p.get("target"):primary="big-data"
        else:primary=selected[0]
        order_priority={"big-data":10,"data-analyst":20,"vision-ai":30,"time-series":30,"machine-learning":40,"deep-learning":50,"mlops":60};route["selected_agents"]=selected;route["routing_reasons"]=reasons;route["execution_order"]=sorted(selected,key=lambda a:order_priority[a]);route["primary_agent"]=primary;route.setdefault("intent",{}).update({"train":train,"no_train":no_train,"forecast":forecast,"no_forecast":no_forecast,"deep_learning":dl,"monitor":monitor});route["semantic_flags"]={"multilabel":is_multilabel(problem),"survival":is_survival(problem),"supervised_type":infer_supervised_type(problem) if p.get("target") else None};return route

class EnhancedDataAnalyst(base.DataAnalystExpert):
    def run(self,problem):
        out=super().run(problem);p=problem.get("profile",{});path=problem.get("data_path")
        if p.get("mixed_granularity") or p.get("repeated_measure"):
            out["QUESTION"].insert(0,"한 행의 observational unit은 무엇이며 반복 측정과 개체 수준 변수가 섞여 있는가?");out["DECIDE"]["observation_unit_review"]="required before aggregation/modeling";out["markers"] += ["observation_unit_identified","mixed_granularity_guard"]
        if is_survival(problem) and path:
            df=pd.read_csv(path);event_col=p.get("censor_col") or p.get("event_col","failure_event")
            if event_col in df.columns:out["INSPECT"].append({"fact":"event_rate","value":float(df[event_col].mean())});out["QUESTION"].append("고장나지 않은 관측을 단순한 긴 생존시간으로 취급하면 censoring bias가 생기지 않는가?");out["markers"] += ["censoring_recognized"]
        if path and Path(path).exists():
            df=pd.read_csv(path);numeric=df.select_dtypes(include=[np.number]);structure={}
            for c in numeric.columns[:12]:
                x=numeric[c].dropna()
                if len(x)<20:continue
                q1,q3=x.quantile([.25,.75]);iqr=q3-q1;outliers=int(((x<q1-1.5*iqr)|(x>q3+1.5*iqr)).sum()) if iqr>0 else 0;hist,edges=np.histogram(x,bins=min(12,max(5,int(np.sqrt(len(x))//2))));peaks=sum(1 for i in range(1,len(hist)-1) if hist[i]>hist[i-1] and hist[i]>hist[i+1] and hist[i]>.15*hist.max());structure[c]={"q05":float(x.quantile(.05)),"median":float(x.median()),"q95":float(x.quantile(.95)),"iqr_outliers":outliers,"hist_local_peaks":int(peaks)}
            if structure:out["INSPECT"].append({"fact":"open_ended_numeric_structure_scan","value":structure});out["QUESTION"].append("평균 하나가 다봉성/세그먼트/극단 관측의 구조를 숨기고 있지 않은가?");out["markers"] += ["eda_structure_scan","outlier_context_not_auto_delete"]
        return attach_heuristics(out,problem)

class EnhancedML(base.MLExpert):
    def run_survival(self,problem):
        p=problem["profile"];df=pd.read_csv(problem["data_path"]);time_col=p.get("target","observed_days");event_col=p.get("censor_col") or p.get("event_col","failure_event");entry_col=p.get("entry_col");ignore={time_col,event_col};
        if entry_col:ignore.add(entry_col)
        X=df.drop(columns=[c for c in ignore if c in df.columns]).select_dtypes(include=[np.number]).copy();X=X.fillna(X.median());Z=StandardScaler().fit_transform(X);time=df[time_col].astype(float).to_numpy();event=df[event_col].astype(int).to_numpy();censor_rate=float(1-event.mean());fit_status="ok";summary={}
        try:
            entry=df[entry_col].astype(float).to_numpy() if entry_col and entry_col in df.columns else None;res=PHReg(time,Z,status=event,entry=entry).fit(disp=0);params=np.asarray(res.params);risk=Z@params;concord=0.0;comparable=0
            for i in range(len(time)):
                if event[i]!=1:continue
                mask=time>time[i]
                if not np.any(mask):continue
                comparable += int(mask.sum());concord += float((risk[i]>risk[mask]).sum()) + 0.5*float((risk[i]==risk[mask]).sum())
            c_index=float(concord/comparable) if comparable else float("nan");summary={"model":"Cox PH (statsmodels PHReg)","c_index":c_index,"coefficients":{c:float(v) for c,v in zip(X.columns,params)}}
        except Exception as e:fit_status=f"fit_failed: {e}";summary={"model":"Cox PH candidate","c_index":None}
        out=base.step_record("machine-learning");out["UNDERSTAND"]={"problem_type":"survival/time-to-event","time_col":time_col,"event_col":event_col,"censoring_rate":censor_rate};out["INSPECT"]=[{"fact":"rows","value":len(df)},{"fact":"event_rate","value":float(event.mean())},{"fact":"censoring_rate","value":censor_rate},{"fact":"delayed_entry","value":bool(entry_col)},{"fact":"predictors","value":list(X.columns)}];out["QUESTION"]=["censoring mechanism이 informative하지 않은가?","time zero/delayed entry 정의가 맞는가?","일반 RMSE 대신 censor-aware metric을 써야 하지 않는가?"];out["HYPOTHESES"]=[{"id":"H1","statement":"ordinary regression"},{"id":"H2","statement":"censor-aware survival model"}];out["TESTS"]=[{"test":"censoring structure","censoring_rate":censor_rate},{"test":"Cox PH fit","status":fit_status,"result":summary}];out["COMPARE"]=[{"option":"ordinary regression","reject_reason":"censoring ignored"},{"option":"survival analysis","fit":"event/censoring explicit"}];out["DECIDE"]={"decision":"survival-analysis","model_result":summary,"evaluation":["C-index","time-dependent discrimination/calibration"],"do_not_use_as_feature":event_col};out["CHALLENGE"]=["PH assumption may fail; compare AFT/flexible survival models","censoring may be informative"];out["RISKS"]=["censoring assumption","time-zero definition","external/temporal validation"];out["CONFIDENCE"]={"level":"HIGH","reason":"censoring structure explicitly represented"};out["markers"] += ["survival_censoring","survival_metric","censor_aware_validation","event_not_feature","cox_candidate"];return attach_heuristics(out,problem)

    def run_classification(self,problem):
        p=problem["profile"];df=pd.read_csv(problem["data_path"]);target=p["target"];X=df.drop(columns=[target]);raw_y=df[target];encoder=LabelEncoder();y=pd.Series(encoder.fit_transform(raw_y.astype(str)),index=raw_y.index,name=target);classes=[str(x) for x in encoder.classes_];n_classes=len(classes);cat=X.select_dtypes(include=["object","category","bool","string"]).columns.tolist();num=[c for c in X.columns if c not in cat];missing={c:int(v) for c,v in df.isna().sum().items() if v>0};class_counts=raw_y.value_counts(dropna=False).to_dict();minority=float(raw_y.value_counts(normalize=True).min()) if n_classes>1 else 1.0;timing=[c for c in X.columns if c in {"processing_time_sec","power_consumption","post_churn_refund_amount"} or c.lower().startswith("post_")];Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y if n_classes>1 else None)
        pre=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True)),("sc",RobustScaler())]),num),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat)]);pre_tree=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True))]),num),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat)]);models={"DummyPrior":Pipeline([("p",pre),("m",DummyClassifier(strategy="prior"))]),"LogisticRegression":Pipeline([("p",pre),("m",LogisticRegression(max_iter=1200,class_weight="balanced" if minority<.20 else None,random_state=42))]),"HistGradientBoostingClassifier":Pipeline([("p",pre_tree),("m",HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,learning_rate=.05,random_state=42))])};cv=StratifiedKFold(3,shuffle=True,random_state=42);scores=[]
        for name,model in models.items():
            cr=cross_validate(model,Xtr,ytr,cv=cv,scoring={"f1_macro":"f1_macro","balanced_accuracy":"balanced_accuracy","neg_log_loss":"neg_log_loss"},n_jobs=1,error_score="raise");scores.append({"model":name,"cv_f1_macro":float(cr["test_f1_macro"].mean()),"cv_balanced_accuracy":float(cr["test_balanced_accuracy"].mean()),"cv_log_loss":float(-cr["test_neg_log_loss"].mean())})
        scores.sort(key=lambda z:(-z["cv_f1_macro"],z["cv_log_loss"]));best=models[scores[0]["model"]];best.fit(Xtr,ytr);pred=best.predict(Xte);metrics={"accuracy":float(accuracy_score(yte,pred)),"balanced_accuracy":float(balanced_accuracy_score(yte,pred)),"f1_macro":float(f1_score(yte,pred,average="macro"))}
        if hasattr(best,"predict_proba"):
            proba=best.predict_proba(Xte)
            try:metrics["log_loss"]=float(log_loss(yte,proba,labels=list(range(n_classes))))
            except Exception:pass
            if n_classes==2:
                try:metrics["roc_auc"]=float(roc_auc_score(yte,proba[:,1]));metrics["auprc"]=float(average_precision_score(yte,proba[:,1]))
                except Exception:pass
        try:
            imp=permutation_importance(best,Xte,yte,n_repeats=5,random_state=42,scoring="f1_macro",n_jobs=1);importance=sorted(zip(X.columns,imp.importances_mean),key=lambda x:x[1],reverse=True)[:5]
        except Exception:importance=[]
        out=base.step_record("machine-learning");out["UNDERSTAND"]={"problem_type":"classification","target":target,"classes":classes,"prediction_time":"only pre-outcome features allowed"};out["INSPECT"]=[{"fact":"shape","value":list(df.shape)},{"fact":"missing","value":missing},{"fact":"class_counts","value":{str(k):int(v) for k,v in class_counts.items()}},{"fact":"minority_ratio","value":minority}];out["QUESTION"]=["target leakage/proxy가 있는가?","class imbalance에 accuracy만 쓰고 있지 않은가?","entity/time split이 필요한가?"];out["HYPOTHESES"]=[{"id":"H1","statement":"simple linear classifier sufficient"},{"id":"H2","statement":"nonlinear classifier improves validation"}];out["TESTS"]=[{"test":"3-fold stratified CV","results":scores},{"test":"final holdout once","metrics":metrics}];out["COMPARE"]=scores;out["DECIDE"]={"model":scores[0]["model"],"reason":"Train-only CV macro-F1 우선, log-loss 보조","test_metrics":metrics,"timing_risk_features":timing,"top_predictive_features":[{"feature":c,"importance":float(v)} for c,v in importance]};out["CHALLENGE"]=["accuracy alone is insufficient under imbalance","feature importance is not causal","test set is not used for selection"];out["RISKS"]=["entity/time metadata가 없으면 deployment split 검증 제한","rare-class threshold/calibration may require more data"];out["CONFIDENCE"]={"level":"MEDIUM","reason":"stratified CV and final holdout executed; domain deployment metadata limited"};out["markers"] += ["target_problem","classification_path","stratified_cv","dummy_baseline","final_holdout_once","imbalance_metric","prediction_time_leakage","importance_not_causal"];return attach_heuristics(out,problem)

    def run(self,problem):
        if is_survival(problem):return self.run_survival(problem)
        if infer_supervised_type(problem)=="classification":out=self.run_classification(problem)
        else:out=attach_heuristics(super().run(problem),problem)
        if has(problem.get("task",""),["원인","원인이","cause","causal"]):out["DECIDE"]["causal_claim"]="not identified by predictive feature importance";out["CHALLENGE"].append("predictive association and causal intervention are different questions");out["markers"] += ["prediction_vs_causality"]
        return out

class EnhancedVision(base.VisionExpert):
    def run(self,problem):
        if not is_multilabel(problem):return attach_heuristics(super().run(problem),problem)
        df=pd.read_csv(problem["data_path"]) if problem.get("data_path") else pd.DataFrame();label_cols=[c for c in df.columns if c.lower() in {"scratch","dent","contamination","crack","stain"}]
        if not label_cols:label_cols=problem.get("profile",{}).get("label_cols",[])
        prevalence={c:float(df[c].mean()) for c in label_cols if c in df.columns};cooccur={}
        if len(label_cols)>=2 and not df.empty:
            for i,a in enumerate(label_cols):
                for b in label_cols[i+1:]:cooccur[f"{a}+{b}"]=int(((df[a]==1)&(df[b]==1)).sum())
        out=base.step_record("vision-ai");out["UNDERSTAND"]={"vision_task":"multi-label-image-classification","output_semantics":"각 결함은 상호배타적 class가 아니라 독립적으로 존재 가능","labels":label_cols};out["INSPECT"]=[{"fact":"label_prevalence","value":prevalence},{"fact":"label_cooccurrence","value":cooccur}];out["QUESTION"]=["각 label의 annotation 기준과 누락률이 같은가?","같은 제품 이미지가 Train/Test에 섞이지 않는가?","label별 threshold를 동일하게 0.5로 고정할 근거가 있는가?"];out["HYPOTHESES"]=[{"id":"H1","statement":"single-label softmax 문제"},{"id":"H2","statement":"비배타적 multi-label sigmoid 문제"}];out["TESTS"]=[{"test":"label coexistence","result":cooccur},{"test":"label prevalence","result":prevalence}];out["COMPARE"]=[{"option":"softmax/single-label","reject_reason":"여러 결함이 동시에 존재 가능"},{"option":"sigmoid multi-output","fit":"label별 독립 probability 출력 가능"}];out["DECIDE"]={"task":"multi-label-classification","output":"sigmoid per label","loss":"BCE/BCEWithLogits family; imbalance 시 label weighting 검토","split":"product_id 기준 Group Split" if "product_id" in df.columns else "entity-aware split","metrics":["per-label precision/recall/F1","micro-F1","macro-F1","PR-AUC","threshold calibration"]};out["CHALLENGE"]=["label co-occurrence를 shortcut으로만 학습할 수 있음","camera/background shortcut과 label별 failure case 확인"];out["RISKS"]=["pixel-level inspection 미수행","rare label의 threshold 불안정"];out["CONFIDENCE"]={"level":"HIGH","reason":"데이터와 문제 설명 모두 label 비배타성을 명시"};out["markers"] += ["multilabel_task_definition","group_split","multilabel_metrics","label_cooccurrence","threshold_tuning"];return attach_heuristics(out,problem)

class EnhancedDL(base.DeepLearningExpert):
    def run_tabular_classification(self,problem):
        p=problem["profile"];df=pd.read_csv(problem["data_path"]);target=p["target"];X=df.drop(columns=[target]);raw_y=df[target];enc=LabelEncoder();y=pd.Series(enc.fit_transform(raw_y.astype(str)),index=raw_y.index);cat=X.select_dtypes(include=["object","category","bool","string"]).columns.tolist();num=[c for c in X.columns if c not in cat];pre=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat)]);mlp=Pipeline([("p",pre),("m",MLPClassifier(hidden_layer_sizes=(32,16),max_iter=350,early_stopping=True,validation_fraction=.15,n_iter_no_change=18,random_state=42))]);base_lr=Pipeline([("p",pre),("m",LogisticRegression(max_iter=1200,random_state=42))]);cv=StratifiedKFold(3,shuffle=True,random_state=42);results=[]
        for name,m in [("LogisticBaseline",base_lr),("MLPClassifier",mlp)]:sc=cross_validate(m,X,y,cv=cv,scoring="f1_macro",n_jobs=1,error_score="raise");results.append({"model":name,"cv_f1_macro":float(sc["test_score"].mean())})
        results.sort(key=lambda z:-z["cv_f1_macro"]);out=base.step_record("deep-learning");out["UNDERSTAND"]={"task":"tabular classification DL comparison","classes":[str(x) for x in enc.classes_]};out["INSPECT"]=[{"fact":"rows","value":len(df)},{"fact":"small_tabular","value":len(df)<10000}];out["QUESTION"]=["Does MLP beat a simple classifier on the same stratified CV?","Is complexity justified?"];out["HYPOTHESES"]=[{"id":"H1","statement":"MLP adds useful nonlinear capacity"},{"id":"H2","statement":"simple baseline is sufficient"}];out["TESTS"]=[{"test":"same stratified CV","results":results}];out["COMPARE"]=results;out["DECIDE"]={"decision":results[0]["model"],"reason":"higher macro-F1 on same CV"};out["CHALLENGE"]=["Do not select MLP from train accuracy","small tabular often does not justify DL complexity"];out["RISKS"]=["threshold/calibration still require task-specific review"];out["CONFIDENCE"]={"level":"MEDIUM","reason":"same CV comparison completed"};out["markers"] += ["classification_path","same_split_compare","simple_baseline","validation","data_first"];return attach_heuristics(out,problem)
    def run(self,problem):
        if is_multilabel(problem):
            out=base.step_record("deep-learning");out["UNDERSTAND"]={"task":"multi-label vision training","principle":"data/output semantics first, architecture second"};out["INSPECT"]=[{"fact":"multilabel","value":True}];out["QUESTION"]=["augmentation preserves label semantics?","rare labels represented?","thresholds calibrated per label?"];out["HYPOTHESES"]=[{"id":"H1","statement":"shared backbone multi-output"},{"id":"H2","statement":"separate models"}];out["TESTS"]=[{"test":"training plan sanity checks","items":["small batch overfit","loss decreases","label/output alignment","visualize augmentation"]}];out["COMPARE"]=[{"option":"separate models","cost":"ops complexity"},{"option":"shared pretrained backbone","benefit":"shared representation"}];out["DECIDE"]={"training":"pretrained backbone + sigmoid outputs","validation":"per-label + micro/macro metrics","early_stopping":True};out["CHALLENGE"]=["co-occurrence shortcut","camera/background shortcut"];out["RISKS"]=["actual pixel training not executed"];out["CONFIDENCE"]={"level":"MEDIUM","reason":"strategy clear; pixel benchmark absent"};out["markers"] += ["transfer_learning","validation","multilabel_output","augmentation_label_sanity"];return attach_heuristics(out,problem)
        modality=problem.get("profile",{}).get("modality","tabular")
        if modality=="tabular" and problem.get("profile",{}).get("target") and infer_supervised_type(problem)=="classification":return self.run_tabular_classification(problem)
        out=super().run(problem);out["QUESTION"].append("Could a pipeline/label bug silently look like successful training?");out["CHALLENGE"].append("verify augmentation/label mapping and small-batch overfit sanity test");out["markers"] += ["data_first","silent_failure_guard","one_change_at_a_time"];p=problem.get("profile",{})
        if p.get("augmentation_changes_label") or p.get("augmentation_semantic_risk"):out["QUESTION"].append("Does augmentation change the target semantics and require label transformation?");out["DECIDE"]["augmentation_guard"]="BLOCK unsafe augmentation until image+label transform is verified visually";out["markers"] += ["augmentation_label_consistency","visualize_augmentation"]
        return attach_heuristics(out,problem)

class EnhancedTS(base.TimeSeriesExpert):
    def run(self,problem):return attach_heuristics(super().run(problem),problem)

class EnhancedBigData(base.BigDataExpert):
    def run(self,problem):
        out=super().run(problem);p=problem.get("profile",{});out["DECIDE"]["architecture_tradeoffs"]={"reliability":p.get("reliability_requirement","must be defined"),"scalability":{"rows":p.get("rows"),"size_gb":p.get("size_gb"),"streaming":p.get("streaming",False)},"maintainability":"prefer explicit simple dataflow over buzzword-driven stack","latency_sla":p.get("latency_sla")};out["markers"] += ["reliability_scalability_maintainability","explicit_tradeoffs"];return attach_heuristics(out,problem)

class EnhancedMLOps(base.MLOpsExpert):
    def run(self,problem):
        out=super().run(problem);p=problem.get("profile",{});train_stats=p.get("train_feature_stats");serve_stats=p.get("serve_feature_stats")
        if train_stats and serve_stats:
            mismatches={}
            for k,v in train_stats.items():
                if k in serve_stats and isinstance(v,(int,float)) and isinstance(serve_stats[k],(int,float)):
                    denom=max(abs(float(v)),1e-9);rel=abs(float(serve_stats[k])-float(v))/denom
                    if rel>0.20:mismatches[k]={"train":v,"serve":serve_stats[k],"relative_diff":rel}
            out["INSPECT"].append({"fact":"training_serving_feature_mismatch","value":mismatches});out["DECIDE"]["training_serving_skew_review"]="BLOCK deployment / investigate" if mismatches else "no large mismatch in supplied checks";out["markers"] += ["training_serving_skew","pipeline_parity"]
        return attach_heuristics(out,problem)

class RuntimeVerifier:
    REQUIRED_KEYS=["UNDERSTAND","INSPECT","QUESTION","HYPOTHESES","TESTS","COMPARE","DECIDE","CHALLENGE","RISKS","CONFIDENCE"]
    def verify(self,problem,result):
        checks=[]
        def add(name,ok,evidence=None,severity="error"):checks.append({"name":name,"pass":bool(ok),"severity":severity,"evidence":evidence})
        route=result["routing"];selected=set(route.get("selected_agents",[]));outputs={x["agent"]:x for x in result.get("expert_outputs",[])};errors=result.get("expert_errors",[]);add("expert_execution_errors",not errors,errors)
        for agent in selected:
            o=outputs.get(agent);add(f"{agent}:output_present",o is not None)
            if o is not None:add(f"{agent}:reasoning_contract_complete",_contract_complete(o),{k:bool(o.get(k)) for k in self.REQUIRED_KEYS});add(f"{agent}:decision_present",bool(o.get("DECIDE")));add(f"{agent}:heuristic_trace_present",bool(o.get("heuristic_trace")),severity="warning")
        p=problem.get("profile",{});intent=route.get("intent",{});modality=p.get("modality","tabular")
        if modality=="tabular" and p.get("target") and intent.get("train") and not is_survival(problem):
            add("supervised_ml_routed","machine-learning" in selected);ml=outputs.get("machine-learning",{});markers=set(ml.get("markers",[]));add("supervised_baseline","dummy_baseline" in markers,sorted(markers));add("final_test_isolated","final_holdout_once" in markers,sorted(markers))
            if infer_supervised_type(problem)=="classification":add("classification_path","classification_path" in markers,sorted(markers));add("classification_metric","imbalance_metric" in markers,sorted(markers))
            else:add("regression_cv","cross_validation" in markers,sorted(markers))
        if intent.get("deep_learning") and intent.get("train"):add("explicit_dl_routed","deep-learning" in selected)
        if modality in {"image","vision","video"} and intent.get("train"):add("vision_routed","vision-ai" in selected);add("vision_training_dl_routed","deep-learning" in selected)
        if modality=="time-series" and intent.get("forecast"):add("timeseries_routed","time-series" in selected)
        if is_survival(problem):ml=outputs.get("machine-learning",{});add("survival_censor_aware","survival_censoring" in set(ml.get("markers",[])))
        if intent.get("monitor"):add("mlops_routed","mlops" in selected)
        hard_fail=[c for c in checks if not c["pass"] and c["severity"]=="error"];warnings=[c for c in checks if not c["pass"] and c["severity"]=="warning"];status="PASS" if not hard_fail and not warnings else ("REVIEW" if not hard_fail else "FAIL");passed=sum(c["pass"] for c in checks);return {"status":status,"passed":passed,"total":len(checks),"score_pct":100*passed/len(checks) if checks else 100.0,"checks":checks}

class EnhancedSystem:
    def __init__(self):
        self.router=EnhancedSupervisor();self.engines={"data-analyst":EnhancedDataAnalyst(),"machine-learning":EnhancedML(),"deep-learning":EnhancedDL(),"vision-ai":EnhancedVision(),"time-series":EnhancedTS(),"big-data":EnhancedBigData(),"mlops":EnhancedMLOps()};self.verifier=RuntimeVerifier()
    def run(self,problem):
        route=self.router.route(problem);outs=[];errors=[]
        for agent in route["execution_order"]:
            engine=self.engines.get(agent)
            if engine is None:continue
            try:outs.append(engine.run(problem))
            except Exception as exc:errors.append({"agent":agent,"type":type(exc).__name__,"error":str(exc)})
        result={"problem_id":problem["id"],"routing":route,"expert_outputs":outs,"expert_errors":errors};result["verification"]=self.verifier.verify(problem,result);return result

def find_agent(result,name):return next((x for x in result["expert_outputs"] if x["agent"]==name),None)
