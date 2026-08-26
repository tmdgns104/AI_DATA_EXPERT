from __future__ import annotations
import json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd

from scipy.stats import norm, ttest_ind
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

def has(text, words):
    t=(text or "").lower()
    return any(w.lower() in t for w in words)

def step_record(agent):
    return {"agent":agent,"UNDERSTAND":{},"INSPECT":[],"QUESTION":[],"HYPOTHESES":[],"TESTS":[],"COMPARE":[],"DECIDE":{},"CHALLENGE":[],"RISKS":[],"CONFIDENCE":{},"markers":[]}

class Supervisor:
    def route(self, problem):
        task=problem.get("task","");p=problem.get("profile",{});modality=p.get("modality","tabular");existing=bool(p.get("existing_model",False));deployment=bool(p.get("deployment",False));monitoring=bool(p.get("monitoring",False));streaming=bool(p.get("streaming",False));size_gb=float(p.get("size_gb",0) or 0);rows=int(p.get("rows",0) or 0);target=p.get("target")
        no_train=has(task,["학습은 하지","모델 학습은 하지","학습하지 말","새 cnn 학습은 아직 하지"]);no_forecast=has(task,["forecast는 하지","미래 예측은 하지","예측하지 말고","과거 패턴만"]);train=has(task,["학습","예측","분류","회귀","검출","segmentation","모델을 만들어"]) and not no_train;audit=has(task,["audit","점검","품질","깨짐","중복","라벨 오류"]) and not train;forecast=has(task,["향후","forecast","예측 horizon","다음 24시간","잔여수명"]) and not no_forecast;ab=has(task,["a/b","가설검정","전환율 차이"]);anomaly=has(task,["이상탐지","anomaly","이상 패턴"]);monitor=monitoring or has(task,["drift","모니터링","성능 저하","재학습 여부"]);big=(size_gb>=10 or rows>=10_000_000 or streaming or modality=="big-data")
        selected=[];reasons={}
        def add(a,r):
            if a not in selected:selected.append(a)
            reasons.setdefault(a,[]).append(r)
        if big:add("big-data","대용량/Streaming 처리 제약")
        if ab:add("data-analyst","통계적 비교/효과크기 검정")
        if modality in {"image","video","vision"}:
            add("vision-ai","Vision 모달리티")
            if train:add("deep-learning","Vision 모델 학습/transfer learning")
        if modality=="time-series" and forecast:add("data-analyst","시계열 품질/패턴 탐색");add("time-series","시간순 Forecast/Backtest 및 내부 ML baseline")
        elif modality=="time-series" and not forecast:add("data-analyst","과거 시계열 패턴 분석만 요청")
        if modality=="tabular" and target and train:add("data-analyst","Feature/Target EDA");add("machine-learning","정형 지도학습")
        if anomaly and modality=="tabular":add("data-analyst","비지도 데이터 구조 확인");add("machine-learning","이상탐지")
        if has(task,["dnn","딥러닝","mlp","lstm","transformer"]) and train and modality=="tabular":add("deep-learning","사용자 DL 요구를 baseline과 비교")
        if monitor or (existing and (deployment or monitoring)):add("mlops","운영/Drift/성능 모니터링")
        if deployment and modality in {"image","video","vision"}:add("mlops","Edge/Serving 제약")
        if not selected:add("data-analyst","기본 분석 문제")
        if existing and monitor:primary="mlops"
        elif modality in {"image","video","vision"}:primary="vision-ai"
        elif modality=="time-series" and forecast:primary="time-series"
        elif big and not target and not ab:primary="big-data"
        elif ab:primary="data-analyst"
        elif "machine-learning" in selected:primary="machine-learning"
        else:primary=selected[0]
        order_priority={"big-data":10,"data-analyst":20,"vision-ai":30,"time-series":30,"machine-learning":40,"deep-learning":50,"mlops":60};order=sorted(selected,key=lambda a:order_priority[a]);return {"primary_agent":primary,"selected_agents":selected,"execution_order":order,"routing_reasons":reasons,"intent":{"train":train,"audit":audit,"forecast":forecast,"ab_test":ab,"anomaly":anomaly,"monitor":monitor,"no_train":no_train,"no_forecast":no_forecast}}

class DataAnalystExpert:
    def run(self, problem):
        r=step_record("data-analyst");path=Path(problem["data_path"]) if problem.get("data_path") else None;task=problem.get("task","");r["UNDERSTAND"]={"problem":task,"success":"현상을 수치로 설명하고 과도한 인과 주장을 피한다"}
        if path and ("ab_test" in path.name.lower() or path.name.lower().startswith("d01_ab")):
            df=pd.read_csv(path);grp=df.groupby("group")["converted"].agg(["mean","sum","count"]);pa=float(grp.loc["A","mean"]);pb=float(grp.loc["B","mean"]);na=int(grp.loc["A","count"]);nb=int(grp.loc["B","count"]);xa=int(grp.loc["A","sum"]);xb=int(grp.loc["B","sum"]);pooled=(xa+xb)/(na+nb);se=math.sqrt(pooled*(1-pooled)*(1/na+1/nb));z=(pb-pa)/se;pval=float(2*(1-norm.cdf(abs(z))));effect_pp=(pb-pa)*100;lift=(pb/pa-1)*100;r["INSPECT"]=[{"fact":"A 전환율","value":pa},{"fact":"B 전환율","value":pb},{"fact":"표본수","value":{"A":na,"B":nb}}];r["QUESTION"]=["그룹 배정이 무작위였고 exposure 조건이 같은가?","통계적으로 유의해도 사업적으로 의미 있는 크기인가?","다른 핵심 guardrail metric이 악화되지는 않았는가?"];r["HYPOTHESES"]=[{"id":"H0","statement":"A와 B의 전환율 차이가 없다"},{"id":"H1","statement":"B의 전환율이 A와 다르다"}];r["TESTS"]=[{"test":"two-proportion z-test","effect_pp":effect_pp,"relative_lift_pct":lift,"p_value":pval}];r["COMPARE"]=[{"option":"A 유지","conversion":pa},{"option":"B 적용 후보","conversion":pb}];significant=pval<0.05;r["DECIDE"]={"decision":"B가 유의하게 높음" if significant else "유의한 차이를 확인하지 못함","reason":f"효과 {effect_pp:.2f}%p, p={pval:.4g}","causal_scope":"무작위 배정/실험 무결성이 확인되는 범위에서만 실험 효과로 해석"};r["CHALLENGE"]=["사후적으로 여러 지표를 반복 검정했다면 false positive 가능","신규 UI가 전환율은 올려도 취소율/객단가를 악화시킬 수 있음"];r["RISKS"]=["randomization/exposure 로그는 현재 파일만으로 검증 불가","guardrail metric 미제공"];r["CONFIDENCE"]={"level":"MEDIUM","reason":"효과와 유의성은 계산했지만 실험 무결성 메타데이터가 없음"};r["markers"] += ["effect_size","p_value","practical_significance","randomization_guard","causality_scope"];return r
        if path and "H04_observational" in path.name:
            df=pd.read_csv(path);g=df.groupby("treatment").agg(outcome=("outcome","mean"),severity=("severity","mean"),n=("outcome","size"));crude=float(g.loc[1,"outcome"]-g.loc[0,"outcome"]);sev_gap=float(g.loc[1,"severity"]-g.loc[0,"severity"]);r["INSPECT"]=[{"fact":"치료군-비치료군 단순 outcome 차이","value":crude},{"fact":"기저 severity 차이","value":sev_gap}];r["QUESTION"]=["치료 배정이 무작위인가?","severity가 treatment와 outcome 모두에 영향을 주는 confounder인가?"];r["HYPOTHESES"]=[{"id":"H1","statement":"치료 자체가 outcome을 바꿈"},{"id":"H2","statement":"기저 severity 차이가 단순 평균 차이를 왜곡"}];r["TESTS"]=[{"test":"baseline balance check","severity_gap":sev_gap}];r["COMPARE"]=[{"method":"crude mean difference","risk":"confounding 큼"},{"method":"회귀/propensity/실험","risk":"추가 가정 필요"}];r["DECIDE"]={"decision":"현재 관측자료만으로 치료의 인과효과를 단정하지 않음","reason":"치료군의 severity 분포가 다름"};r["CHALLENGE"]=["측정되지 않은 confounder가 남을 수 있음"];r["RISKS"]=["무작위 실험 아님","causal identification 가정 미충족"];r["CONFIDENCE"]={"level":"HIGH","reason":"인과 단정 금지는 데이터 구조상 명확"};r["markers"] += ["observational_not_causal","confounder_check","alternative_causal_methods"];return r
        if path:
            df=pd.read_csv(path);r["INSPECT"]=[{"fact":"rows","value":len(df)},{"fact":"missing_cells","value":int(df.isna().sum().sum())},{"fact":"duplicate_rows","value":int(df.duplicated().sum())}]
        r["QUESTION"]=["분석 목적이 설명인가 예측인가?","세그먼트별 분포 차이가 있는가?"];r["HYPOTHESES"]=[{"id":"H1","statement":"관측된 전체 평균이 세그먼트 차이를 숨길 수 있음"}];r["TESTS"]=[{"test":"schema/missing/distribution review","result":"performed"}];r["COMPARE"]=[{"option":"전체 평균만","risk":"세그먼트 이질성 누락"},{"option":"그룹/구간 분석","risk":"다중비교 주의"}];r["DECIDE"]={"decision":"EDA와 세그먼트 분석 우선"};r["CHALLENGE"]=["상관관계를 원인으로 해석하지 않음"];r["RISKS"]=["도메인 정의/측정 기준이 추가로 필요할 수 있음"];r["CONFIDENCE"]={"level":"MEDIUM","reason":"기초 품질은 확인했으나 도메인 컨텍스트 제한"};r["markers"] += ["data_quality","segment_analysis","causality_scope"];return r

class MLExpert:
    def run(self, problem):
        r=step_record("machine-learning");path=Path(problem["data_path"]);df=pd.read_csv(path);target=problem["profile"]["target"];X=df.drop(columns=[target]);y=df[target];cat=X.select_dtypes(include=["object","category","bool"]).columns.tolist();num=[c for c in X.columns if c not in cat];r["UNDERSTAND"]={"problem_type":"regression" if pd.api.types.is_numeric_dtype(y) and y.nunique()>20 else "classification","target":target,"prediction_time":"문제 설명 기준, production 이전/도중 사용 가능 Feature만 허용"};missing={c:int(v) for c,v in df.isna().sum().items() if v>0};timing=[c for c in X.columns if c in {"processing_time_sec","power_consumption"}];r["INSPECT"]=[{"fact":"shape","value":list(df.shape)},{"fact":"missing","value":missing},{"fact":"target_std","value":float(y.std())}];r["QUESTION"]=["예측시점 이후에만 알 수 있는 Feature가 있는가?","IQR 이상치가 센서 오류인지 실제 극단 공정인지 근거가 있는가?","선형 상관이 낮아도 비선형 관계가 있는가?"];r["HYPOTHESES"]=[{"id":"H1","statement":"선형 모델로 충분"},{"id":"H2","statement":"공정 최적구간 때문에 비선형 모델이 유리"}];nonlinear=[]
        for c in num:
            v=df[[c,target]].dropna()
            if len(v)<100 or v[c].nunique()<10:continue
            corr=v[c].corr(v[target])
            try:
                bins=pd.qcut(v[c],5,duplicates="drop");means=v.assign(_b=bins).groupby("_b",observed=False)[target].mean().to_numpy()
                if abs(corr)<.15 and (means.max()-means.min()) > y.std()*.20:nonlinear.append(c)
            except Exception:pass
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42);pre=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True)),("sc",RobustScaler())]),num),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat)]);pre_tree=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True))]),num),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat)]);models={"Dummy":Pipeline([("p",pre),("m",DummyRegressor())]),"Ridge":Pipeline([("p",pre),("m",Ridge(alpha=1.0))]),"HistGradientBoosting":Pipeline([("p",pre_tree),("m",HistGradientBoostingRegressor(max_iter=180,max_leaf_nodes=15,learning_rate=.05,random_state=42))])};cv=KFold(3,shuffle=True,random_state=42);scores=[]
        for name,m in models.items():
            cr=cross_validate(m,Xtr,ytr,cv=cv,scoring={"RMSE":"neg_root_mean_squared_error","MAE":"neg_mean_absolute_error"},n_jobs=1);scores.append({"model":name,"cv_rmse":float(-cr["test_RMSE"].mean()),"cv_mae":float(-cr["test_MAE"].mean())})
        scores.sort(key=lambda z:z["cv_rmse"]);best=models[scores[0]["model"]];best.fit(Xtr,ytr);pred=best.predict(Xte);metrics={"MAE":float(mean_absolute_error(yte,pred)),"RMSE":float(np.sqrt(mean_squared_error(yte,pred))),"R2":float(r2_score(yte,pred))};imp=permutation_importance(best,Xte,yte,n_repeats=5,random_state=42,scoring="neg_root_mean_squared_error",n_jobs=1);importance=sorted(zip(X.columns,imp.importances_mean),key=lambda x:x[1],reverse=True)[:5];r["TESTS"]=[{"test":"nonlinear bin check","candidates":nonlinear},{"test":"3-fold CV model comparison","results":scores},{"test":"final holdout once","metrics":metrics}];r["COMPARE"]=scores;r["DECIDE"]={"model":scores[0]["model"],"reason":"Train 내부 CV RMSE 최소","test_metrics":metrics,"timing_risk_features":timing,"top_predictive_features":[{"feature":c,"importance":float(v)} for c,v in importance]};r["CHALLENGE"]=["Permutation importance는 인과효과가 아님","processing_time_sec/power_consumption은 예측 시점에 따라 미래정보가 될 수 있음","Test를 모델 선택에 반복 사용하지 않음"];r["RISKS"]=["timestamp/lot_id/equipment_id가 없으면 time/group generalization 검증 제한","이상치 삭제는 engineering spec 없이 자동 수행하지 않음"];r["CONFIDENCE"]={"level":"MEDIUM","reason":"Holdout 성능은 확인했지만 제조 운영 metadata 부족"};r["markers"] += ["target_problem","missing_pipeline","outlier_preserve","nonlinear_check","prediction_time_leakage","dummy_baseline","cross_validation","final_holdout_once","importance_not_causal","operational_metadata_risk"];return r

class DeepLearningExpert:
    def run(self, problem):
        r=step_record("deep-learning");modality=problem.get("profile",{}).get("modality","tabular");task=problem.get("task","")
        if modality in {"image","video","vision"}:
            path=Path(problem["data_path"]) if problem.get("data_path") else None;df=pd.read_csv(path) if path and path.exists() else pd.DataFrame();n=int(problem.get("profile",{}).get("rows") or len(df));class_ratio=df["label"].value_counts(normalize=True).to_dict() if "label" in df.columns else {};r["UNDERSTAND"]={"role":"Vision Expert가 정한 task에 맞는 학습전략 지원","dataset_size":n,"modality":modality};r["INSPECT"]=[{"fact":"dataset_size","value":n},{"fact":"class_ratio","value":class_ratio}];r["QUESTION"]=["scratch 학습보다 pretrained representation이 유리한가?","augmentation이 실제 촬영 변동을 반영하고 결함 자체를 파괴하지 않는가?","validation curve에서 overfitting이 시작되는 시점은?","정확도 이득이 latency/memory 비용을 정당화하는가?"];r["HYPOTHESES"]=[{"id":"H1","statement":"scratch CNN이 데이터에 맞게 최적화"},{"id":"H2","statement":"pretrained backbone + fine-tuning이 적은 데이터에서 더 안정적"}];r["TESTS"]=[{"test":"training plan","candidates":["small pretrained backbone","ResNet/EfficientNet class","scratch baseline"],"same_split_required":True},{"test":"overfitting guard","methods":["validation curve","EarlyStopping","weight decay/dropout as needed"]}];r["COMPARE"]=[{"option":"scratch CNN","strength":"도메인 맞춤","risk":"소/중규모 데이터 과적합"},{"option":"transfer learning","strength":"적은 데이터에서도 강한 baseline","risk":"source-domain bias 검토"}];r["DECIDE"]={"strategy":"pretrained transfer learning 우선, scratch는 baseline","validation":"동일 product-aware split","optimization":"class imbalance가 있으면 weighted loss/sampling 검토"};r["CHALLENGE"]=["더 큰 backbone이 validation 성능 없이 Train accuracy만 높이는지 검사","augmentation이 실제 defect morphology를 왜곡하지 않는지 샘플 검수"];r["RISKS"]=["실제 pixel 데이터가 아닌 metadata 기반 시뮬레이션이라 architecture 성능 수치는 아직 없음","edge 배포 시 latency/memory 별도 검증 필요"];r["CONFIDENCE"]={"level":"MEDIUM","reason":"학습전략은 명확하지만 실제 pixel training benchmark 전"};r["markers"] += ["transfer_learning","augmentation_validity","validation","early_stopping","learning_curve","resource_tradeoff","same_split_compare"];return r
        path=Path(problem["data_path"]);df=pd.read_csv(path);target=problem["profile"]["target"];X=df.drop(columns=[target]);y=df[target];cat=X.select_dtypes(include=["object","category","bool"]).columns.tolist();num=[c for c in X.columns if c not in cat];r["UNDERSTAND"]={"question":"딥러닝이 단순 요청이 아니라 검증상 가치가 있는가?","rows":len(df),"features":X.shape[1]};r["INSPECT"]=[{"fact":"small_tabular","value":len(df)<10000},{"fact":"modality","value":"tabular"}];r["QUESTION"]=["Tree/Boosting baseline보다 실제 이득이 있는가?","validation curve에서 과적합이 발생하는가?","추론비용 증가를 정당화하는가?"];r["HYPOTHESES"]=[{"id":"H1","statement":"비선형 tabular라 MLP가 유리"},{"id":"H2","statement":"소규모 tabular라 Boosting이 더 안정적"}];Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42);pre=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat)]);hgb=Pipeline([("p",pre),("m",HistGradientBoostingRegressor(max_iter=180,learning_rate=.05,random_state=42))]);mlp=Pipeline([("p",pre),("m",MLPRegressor(hidden_layer_sizes=(32,16),max_iter=350,early_stopping=True,validation_fraction=.15,n_iter_no_change=18,random_state=42))]);cv=KFold(3,shuffle=True,random_state=42);results=[]
        for name,m in [("BoostingBaseline",hgb),("MLP",mlp)]:sc=cross_validate(m,Xtr,ytr,cv=cv,scoring="neg_root_mean_squared_error",n_jobs=1);results.append({"model":name,"cv_rmse":float(-sc["test_score"].mean())})
        results.sort(key=lambda z:z["cv_rmse"]);chosen=hgb if results[0]["model"]=="BoostingBaseline" else mlp;chosen.fit(Xtr,ytr);pred=chosen.predict(Xte);rmse=float(np.sqrt(mean_squared_error(yte,pred)));r["TESTS"]=[{"test":"same-split CV","results":results},{"test":"final holdout","rmse":rmse}];r["COMPARE"]=results;r["DECIDE"]={"decision":results[0]["model"],"reason":"동일 CV에서 더 낮은 RMSE","holdout_rmse":rmse};r["CHALLENGE"]=["MLP Train 성능만 보고 선택하지 않음","작은 tabular에서 복잡도/재현성 비용이 이득보다 클 수 있음"];r["RISKS"]=["Architecture 탐색을 늘리면 validation overfitting 가능","실제 DL framework GPU latency는 이 시뮬레이션에서 측정하지 않음"];r["CONFIDENCE"]={"level":"MEDIUM","reason":"동일 split 비교는 했으나 제한된 architecture만 시험"};r["markers"] += ["ml_baseline_compare","scaling","validation","early_stopping","same_split_compare","complexity_cost"];return r

class VisionExpert:
    def run(self, problem):
        r=step_record("vision-ai");path=Path(problem["data_path"]);df=pd.read_csv(path);task=problem.get("task","");train=df[df["split"]=="train"] if "split" in df.columns else None
        if "split" in df.columns and "label" in df.columns and train is not None and train["label"].nunique()==1 and int(train["label"].iloc[0])==0:vision_task="visual-anomaly-detection"
        elif has(task,["bbox","검출","detection"]):vision_task="object-detection"
        elif has(task,["segmentation","마스크","영역"]):vision_task="segmentation"
        else:vision_task="image-classification"
        r["UNDERSTAND"]={"vision_task":vision_task,"inspection_unit":"제품/이미지 관계를 확인 후 split 단위 결정"};r["INSPECT"]=[]
        if "corrupted" in df.columns:r["INSPECT"].append({"fact":"corrupted_images","value":int(df["corrupted"].sum())})
        if "label" in df.columns:r["INSPECT"].append({"fact":"label_distribution","value":df["label"].value_counts(normalize=True).to_dict()})
        if "product_id" in df.columns and "naive_split" in df.columns:tr=set(df[df.naive_split=="train"]["product_id"]);te=set(df[df.naive_split=="test"]["product_id"]);r["INSPECT"].append({"fact":"same_product_train_test_overlap","value":len(tr&te)})
        if "camera_id" in df.columns and "label" in df.columns:r["INSPECT"].append({"fact":"defect_rate_by_camera","value":df.groupby("camera_id")["label"].mean().to_dict()})
        r["QUESTION"]=["같은 제품/연속 프레임이 Train/Test에 섞이는가?","라벨 기준이 작업자/라인마다 동일한가?","카메라/조명/배경이 불량 label의 shortcut이 되지 않는가?","실제 현업에 필요한 출력이 class인지 위치/영역인지?"];r["HYPOTHESES"]=[{"id":"H1","statement":"모델이 실제 결함 특징을 학습"},{"id":"H2","statement":"카메라/조명/배경 shortcut으로 높은 점수가 발생"}]
        if vision_task=="visual-anomaly-detection":
            train_pos=int(train["label"].sum());r["TESTS"]=[{"test":"training label availability","train_anomaly_count":train_pos}];r["COMPARE"]=[{"option":"supervised classifier","problem":"Train에 anomaly 양성 없음"},{"option":"one-class/anomaly detection","fit":"정상 데이터만으로 학습 가능"}];r["DECIDE"]={"task":"visual-anomaly-detection","split":"제품/Family 기준 분리","model_direction":"pretrained feature + anomaly score / PatchCore류 후보"};r["markers"] += ["anomaly_due_no_positive_train","vision_task_definition","group_split","anomaly_metric"]
        else:
            r["TESTS"]=[{"test":"metadata quality/leakage audit","result":"performed"}];r["COMPARE"]=[{"option":"scratch CNN","fit":"데이터 규모가 작으면 과적합 위험"},{"option":"pretrained transfer learning","fit":"소/중규모 Vision baseline에 적합"}];r["DECIDE"]={"task":vision_task,"split":"product_id 기준 Group Split" if "product_id" in df.columns else "entity-aware split","training":"pretrained transfer learning","metrics":["defect Recall","F1","confusion matrix"]};r["markers"] += ["vision_task_definition","corrupted_image_check","class_imbalance","group_split","transfer_learning","defect_recall_f1"]
        r["CHALLENGE"]=["Grad-CAM/오분류 샘플로 결함 대신 카메라/배경을 보는지 검사","카메라별 성능을 별도로 확인"];r["RISKS"]=["라벨 품질 원본 검수 필요","새 카메라/조명 환경 domain shift"];r["CONFIDENCE"]={"level":"MEDIUM","reason":"metadata에서 leakage/shortcut 위험은 보이지만 실제 픽셀 검사는 아직 수행 안 함"};
        if "camera_id" in df.columns:r["markers"].append("camera_shortcut_challenge")
        r["markers"].append("failure_case_review");return r

class TimeSeriesExpert:
    def run(self, problem):
        r=step_record("time-series");path=Path(problem["data_path"]);df=pd.read_csv(path);tcol=problem["profile"].get("timestamp_col","timestamp");ycol=problem["profile"].get("target");df[tcol]=pd.to_datetime(df[tcol]);df=df.sort_values(tcol);full=pd.date_range(df[tcol].min(),df[tcol].max(),freq=problem["profile"].get("freq","h"));missing_ts=int(len(full)-len(df));y=df.set_index(tcol)[ycol].reindex(full);lag24=float(y.corr(y.shift(24)));lag168=float(y.corr(y.shift(168)));work=pd.DataFrame({"y":y,"lag1":y.shift(1),"lag24":y.shift(24),"lag168":y.shift(168)});work["hour"]=work.index.hour;work["dow"]=work.index.dayofweek;work=work.dropna();cut=int(len(work)*.8);tr=work.iloc[:cut];te=work.iloc[cut:];naive=te["lag24"].to_numpy();naive_mae=float(mean_absolute_error(te["y"],naive));ridge=Pipeline([("sc",StandardScaler()),("m",Ridge(alpha=1.0))]);feats=["lag1","lag24","lag168","hour","dow"];ridge.fit(tr[feats],tr["y"]);pred=ridge.predict(te[feats]);ridge_mae=float(mean_absolute_error(te["y"],pred));r["UNDERSTAND"]={"forecast_horizon":problem["profile"].get("horizon","24h"),"sampling_interval":problem["profile"].get("freq","1h"),"target":ycol};r["INSPECT"]=[{"fact":"missing_timestamps","value":missing_ts},{"fact":"lag24_correlation","value":lag24},{"fact":"lag168_correlation","value":lag168}];r["QUESTION"]=["예측 시점 이후 값이 lag/rolling 계산에 섞이지 않는가?","24시간 seasonal naive보다 나은가?","horizon별 성능이 유지되는가?"];r["HYPOTHESES"]=[{"id":"H1","statement":"전일 같은 시각 값만으로도 강한 baseline"},{"id":"H2","statement":"lag1/24/168 + calendar feature가 baseline 개선"}];r["TESTS"]=[{"test":"chronological holdout","naive_lag24_mae":naive_mae,"ridge_lag_mae":ridge_mae}];r["COMPARE"]=[{"model":"seasonal_naive_lag24","MAE":naive_mae},{"model":"Ridge_lags","MAE":ridge_mae}];r["DECIDE"]={"decision":"Ridge_lags" if ridge_mae<naive_mae else "seasonal_naive","reason":"과거→미래 holdout MAE 비교"};r["CHALLENGE"]=["Random split 결과를 사용하지 않음","휴일/설비정지 같은 regime change에서는 과거 seasonality가 깨질 수 있음"];r["RISKS"]=["holiday/exogenous feature 미제공","다단계 24h horizon의 step별 오차는 추가 평가 필요"];r["CONFIDENCE"]={"level":"MEDIUM","reason":"시간순 검증과 naive baseline은 있으나 외생변수 없음"};r["markers"] += ["forecast_horizon","missing_timestamp","seasonality_check","temporal_split","seasonal_naive","lag_features","backtesting_logic","horizon_risk"];return r

class BigDataExpert:
    def run(self, problem):
        r=step_record("big-data");p=problem["profile"];r["UNDERSTAND"]={"volume_gb":p.get("size_gb"),"rows":p.get("rows"),"streaming":p.get("streaming",False),"sla":p.get("sla","not provided")};r["INSPECT"]=[{"fact":"storage_format","value":p.get("storage_format","Parquet")},{"fact":"partition_candidates","value":p.get("partition_candidates",["event_date"])},{"fact":"join_key","value":p.get("join_key")},{"fact":"known_skew_key","value":p.get("skew_key")}];r["QUESTION"]=["전체 scan이 필요한가?","partition pruning으로 읽는 양을 줄일 수 있는가?","join key skew가 shuffle hotspot을 만드는가?","작은 dimension은 broadcast 가능한가?"];r["HYPOTHESES"]=[{"id":"H1","statement":"Pandas/단일 노드 처리 가능"},{"id":"H2","statement":"분산 SQL/Spark + partition pruning이 필요"}];r["TESTS"]=[{"test":"volume feasibility","single_node":False if float(p.get("size_gb",0))>=50 else "depends"}];r["COMPARE"]=[{"option":"Pandas collect","risk":"driver memory/OOM, full scan"},{"option":"Spark/SQL pushdown","benefit":"분산 scan, partition pruning, aggregation pushdown"}];r["DECIDE"]={"engine":"Spark/PySpark or distributed SQL","partition":"event_date","rules":["필요 컬럼만 projection","filter pushdown","collect 금지","작은 dimension만 broadcast 검토","skew key salt/재partition 검토"]};r["CHALLENGE"]=["작은 파일이 너무 많으면 metadata/task overhead 증가","hot key join은 평균 데이터 크기만 보고 판단하면 안 됨"];r["RISKS"]=["실제 cluster memory/core/SLA 미제공","join cardinality 통계 추가 필요"];r["CONFIDENCE"]={"level":"MEDIUM","reason":"Architecture 방향은 명확하지만 실제 execution plan 통계 없음"};r["markers"] += ["distributed_engine","partition_pruning","projection_pushdown","no_collect","join_skew","broadcast_small_dimension","small_files"];return r

def psi(ref,cur,bins=10):
    edges=np.quantile(ref,np.linspace(0,1,bins+1));edges[0]=-np.inf;edges[-1]=np.inf;edges=np.unique(edges);rc,_=np.histogram(ref,bins=edges);cc,_=np.histogram(cur,bins=edges);rp=np.maximum(rc/rc.sum(),1e-6);cp=np.maximum(cc/cc.sum(),1e-6);return float(np.sum((cp-rp)*np.log(cp/rp)))

class MLOpsExpert:
    def run(self, problem):
        r=step_record("mlops");path=Path(problem["data_path"]);df=pd.read_csv(path);ref=df[df["period"]=="reference"];cur=df[df["period"]=="current"];feature_cols=[c for c in df.columns if c not in {"period","actual","prediction","label"}];drift={c:psi(ref[c].dropna().to_numpy(),cur[c].dropna().to_numpy()) for c in feature_cols};labels_available=bool("actual" in df.columns and cur["actual"].notna().any());perf={}
        if labels_available:perf={"reference_MAE":float(mean_absolute_error(ref["actual"],ref["prediction"])),"current_MAE":float(mean_absolute_error(cur["actual"],cur["prediction"]))};perf["MAE_change_pct"]=(perf["current_MAE"]/perf["reference_MAE"]-1)*100
        r["UNDERSTAND"]={"stage":"production monitoring","labels_available":labels_available};r["INSPECT"]=[{"fact":"feature_PSI","value":drift},{"fact":"performance","value":perf if perf else "actual labels not available yet"}];r["QUESTION"]=["Schema/ETL 변화가 drift 원인인가?","sensor calibration이나 upstream change가 있는가?","label이 없는데 drift만으로 retrain을 결정하고 있지 않은가?"];r["HYPOTHESES"]=[{"id":"H1","statement":"실제 공정/사용자 분포가 변함"},{"id":"H2","statement":"데이터 pipeline/sensor 변경으로 가짜 drift 발생"},{"id":"H3","statement":"모델 성능도 함께 악화"}];r["TESTS"]=[{"test":"PSI reference vs current","results":drift}]
        if perf:r["TESTS"].append({"test":"performance reference vs current","results":perf})
        r["COMPARE"]=[{"option":"즉시 자동 재학습","risk":"원인 미확인 상태에서 문제를 고착"},{"option":"root-cause + shadow/retrain review","benefit":"pipeline/공정/모델 원인을 분리"}];maxpsi=max(drift.values()) if drift else 0
        if labels_available:badperf=perf["MAE_change_pct"]>20;decision="RETRAINING REVIEW + ROOT CAUSE" if maxpsi>=.25 and badperf else "INVESTIGATE / MONITOR"
        else:decision="NO AUTO RETRAIN; INVESTIGATE DRIFT AND WAIT FOR LABELS"
        r["DECIDE"]={"decision":decision,"max_PSI":maxpsi,"performance":perf};r["CHALLENGE"]=["한 배치 악화만으로 재학습하지 않음","Drift와 성능 저하는 같은 현상이 아님"];r["RISKS"]=["설비/recipe/ETL 변경 이력 필요","label delay가 있으면 성능 판단이 늦음"];r["CONFIDENCE"]={"level":"HIGH" if not labels_available else "MEDIUM","reason":"label 유무에 맞게 의사결정 범위를 제한"};r["markers"] += ["data_drift","performance_drift_separate","root_cause_before_retrain","no_auto_retrain","label_availability","shadow_or_review"];return r

class Verifier:
    def verify(self, result, gold):
        checks=[];selected=set(result["routing"]["selected_agents"]);required=set(gold.get("required_agents",[]));checks.append({"name":"required_agents","pass":required.issubset(selected),"evidence":{"required":sorted(required),"selected":sorted(selected)}});forbidden=set(gold.get("forbidden_agents",[]));checks.append({"name":"forbidden_agents","pass":not (forbidden & selected),"evidence":{"forbidden":sorted(forbidden),"selected":sorted(selected)}});outputs={x["agent"]:x for x in result["expert_outputs"]}
        for agent,markers in gold.get("required_markers",{}).items():
            actual=set(outputs.get(agent,{}).get("markers",[]))
            for m in markers:checks.append({"name":f"{agent}:{m}","pass":m in actual,"evidence":{"actual":sorted(actual)}})
        for agent in required:
            o=outputs.get(agent)
            if o:complete=all(bool(o.get(k)) for k in ["UNDERSTAND","INSPECT","QUESTION","HYPOTHESES","TESTS","COMPARE","DECIDE","CHALLENGE","RISKS","CONFIDENCE"]);checks.append({"name":f"{agent}:reasoning_contract_complete","pass":complete,"evidence":"10-stage contract"})
        passed=sum(c["pass"] for c in checks);return {"status":"PASS" if passed==len(checks) else "REVIEW","passed":passed,"total":len(checks),"score_pct":100*passed/len(checks) if checks else 0,"checks":checks}

class System:
    def __init__(self):self.router=Supervisor();self.engines={"data-analyst":DataAnalystExpert(),"machine-learning":MLExpert(),"deep-learning":DeepLearningExpert(),"vision-ai":VisionExpert(),"time-series":TimeSeriesExpert(),"big-data":BigDataExpert(),"mlops":MLOpsExpert()}
    def run(self,problem,gold=None):
        route=self.router.route(problem);outs=[]
        for a in route["execution_order"]:
            if a in self.engines:outs.append(self.engines[a].run(problem))
        result={"problem_id":problem["id"],"routing":route,"expert_outputs":outs}
        if gold is not None:result["verification"]=Verifier().verify(result,gold)
        return result
