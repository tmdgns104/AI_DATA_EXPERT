from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import math
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


INTENT_PROTOTYPES = {
    "ANALYZE_ONLY": [
        "analyze explore summarize inspect data without training",
        "eda exploratory data analysis distribution missing outlier segment",
        "데이터 분석 탐색 요약 품질 결측 이상치 분포 모델 학습 없이",
    ],
    "AUDIT_DATA": [
        "audit dataset labels duplicates leakage quality corruption integrity",
        "review data quality labels duplicate records leakage only",
        "데이터셋 품질 감사 라벨 오류 중복 누수 점검",
    ],
    "TRAIN_MODEL": [
        "train fit build predictive model estimate infer target from features",
        "predict classify regress score estimate output using sensor readings",
        "모델 학습 예측 분류 회귀 추정 센서로 목표값 구하기",
    ],
    "COMPARE_MODELS": [
        "compare models baselines benchmark choose best model validation",
        "baseline versus model compare algorithms select model",
        "모델 비교 기준선 벤치마크 최적 모델 선택",
    ],
    "FORECAST": [
        "forecast future next horizon time series demand load prediction backtest",
        "predict next hours days chronological seasonal naive",
        "시계열 미래 예측 향후 수요 예측 백테스트 계절성",
    ],
    "MONITOR_EXISTING_MODEL": [
        "monitor deployed existing model drift degradation production performance retraining review",
        "training serving skew model monitoring current reference",
        "운영 모델 모니터링 드리프트 성능 저하 재학습 검토",
    ],
    "DEPLOY_MODEL": [
        "deploy serve production inference latency memory onnx tensorrt edge",
        "productionize model serving throughput sla",
        "모델 배포 서빙 운영 추론 지연시간 메모리 온디바이스",
    ],
    "REALTIME_PIPELINE": [
        "streaming realtime pipeline event processing low latency kafka spark",
        "real time event stream architecture",
        "실시간 스트리밍 파이프라인 이벤트 처리",
    ],
    "CAUSAL_ANALYSIS": [
        "causal effect cause treatment intervention confounding experiment",
        "does x cause y estimate treatment effect",
        "인과 효과 원인 처치 개입 교란 실험",
    ],
    "SURVIVAL_ANALYSIS": [
        "survival censored time to event hazard failure risk delayed entry",
        "time until failure censoring event indicator",
        "생존 분석 검열 고장 시간 위험 delayed entry",
    ],
    "VISION_TRAIN": [
        "train image classifier object detection segmentation anomaly vision cnn",
        "computer vision training images defects bounding boxes masks",
        "이미지 분류 객체 탐지 세그멘테이션 비전 학습 결함",
    ],
    "DL_TRAIN": [
        "deep learning neural network dnn mlp cnn lstm transformer pytorch tensorflow training",
        "train neural net gpu small batch overfit checkpoint",
        "딥러닝 신경망 dnn cnn lstm 트랜스포머 pytorch 학습",
    ],
    "ARCHITECTURE": [
        "design data architecture big data distributed storage compute reliable scalable maintainable",
        "spark warehouse lake streaming batch architecture",
        "빅데이터 아키텍처 분산 처리 저장소 확장성 신뢰성",
    ],
}

NEGATION_PATTERNS = {
    "TRAIN_MODEL": [
        r"do\s+not\s+(?:train|fit|build).*model",
        r"don't\s+(?:train|fit|build).*model",
        r"without\s+(?:training|fitting)",
        r"no\s+training",
        r"학습(?:은|을)?\s*(?:하지|금지|말)",
        r"모델\s*학습(?:은|을)?\s*(?:하지|금지|말)",
        r"학습하지\s*마",
    ],
    "FORECAST": [
        r"do\s+not\s+forecast",
        r"don't\s+forecast",
        r"no\s+forecast",
        r"미래\s*예측(?:은|을)?\s*(?:하지|금지|말)",
        r"forecast(?:는|은|을)?\s*(?:하지|금지|말)",
        r"예측하지\s*말고",
    ],
    "MONITOR_EXISTING_MODEL": [],
}


@dataclass
class IntentDecision:
    primary_intent: str
    intents: list[str]
    scores: dict[str, float]
    negated: list[str]
    evidence: list[str]
    confidence: float


class IntentClassifier:
    """Intent-first deterministic classifier.

    It deliberately avoids an external LLM dependency. It combines explicit profile
    signals, negation handling, and TF-IDF similarity against bilingual intent prototypes.
    This is more robust than a flat keyword router but is still a lightweight semantic
    approximation, not a foundation-model semantic parser.
    """

    def __init__(self):
        names: list[str] = []
        docs: list[str] = []
        for name, prototypes in INTENT_PROTOTYPES.items():
            for text in prototypes:
                names.append(name)
                docs.append(text)
        self._names = names
        self._docs = docs
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            lowercase=True,
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(docs)

    def classify(self, task: str, profile: dict[str, Any] | None = None) -> IntentDecision:
        p = profile or {}
        text = (task or "").strip().lower()
        q = self._vectorizer.transform([text or "empty request"])
        sims = cosine_similarity(q, self._matrix)[0]
        scores = {name: 0.0 for name in INTENT_PROTOTYPES}
        for name, score in zip(self._names, sims):
            scores[name] = max(scores[name], float(score))

        evidence: list[str] = []
        negated: list[str] = []
        for intent, patterns in NEGATION_PATTERNS.items():
            if any(re.search(pattern, text, flags=re.I) for pattern in patterns):
                scores[intent] = 0.0
                negated.append(intent)
                evidence.append(f"negation:{intent}")

        modality = str(p.get("modality", "tabular")).lower()
        target = p.get("target")
        if p.get("monitoring") or p.get("existing_model") and (p.get("deployment") or "drift" in text):
            scores["MONITOR_EXISTING_MODEL"] = max(scores["MONITOR_EXISTING_MODEL"], 0.95)
            evidence.append("profile:existing_model/monitoring")
        if p.get("deployment"):
            scores["DEPLOY_MODEL"] = max(scores["DEPLOY_MODEL"], 0.90)
            evidence.append("profile:deployment")
        if p.get("streaming") or float(p.get("size_gb", 0) or 0) >= 10 or int(p.get("rows", 0) or 0) >= 10_000_000:
            scores["ARCHITECTURE"] = max(scores["ARCHITECTURE"], 0.88)
            if p.get("streaming"):
                scores["REALTIME_PIPELINE"] = max(scores["REALTIME_PIPELINE"], 0.92)
            evidence.append("profile:scale/streaming")
        if modality in {"image", "vision", "video"}:
            scores["VISION_TRAIN"] = max(scores["VISION_TRAIN"], 0.55)
            evidence.append("profile:vision_modality")
        if modality == "time-series" and "FORECAST" not in negated and any(k in text for k in ["future", "next", "forecast", "향후", "다음", "horizon"]):
            scores["FORECAST"] = max(scores["FORECAST"], 0.88)
            evidence.append("profile:timeseries+horizon")
        if p.get("censor_col") or p.get("event_col") or p.get("entry_col"):
            scores["SURVIVAL_ANALYSIS"] = max(scores["SURVIVAL_ANALYSIS"], 0.98)
            evidence.append("profile:censor/event")
        if target and "TRAIN_MODEL" not in negated:
            target_language = [
                "estimate", "infer", "derive", "approximate", "score", "calculate", "output",
                "예측", "추정", "구해", "산출", "판별", "분류", "회귀", "맞춰",
            ]
            if any(k in text for k in target_language) or scores["TRAIN_MODEL"] >= 0.08:
                scores["TRAIN_MODEL"] = max(scores["TRAIN_MODEL"], 0.82)
                evidence.append("profile:target+estimation_intent")
        if any(k in text for k in ["dnn", "deep learning", "neural", "cnn", "lstm", "transformer", "딥러닝", "신경망"]):
            scores["DL_TRAIN"] = max(scores["DL_TRAIN"], 0.95)
            evidence.append("explicit:deep_learning")
        if modality not in {"image", "vision", "video"} and not any(k in text for k in ["image", "vision", "이미지", "영상", "object detection", "segmentation"]):
            scores["VISION_TRAIN"] = 0.0
        if modality != "time-series" and not any(k in text for k in ["forecast", "future", "next hour", "next day", "향후", "시계열", "horizon"]):
            scores["FORECAST"] = min(scores["FORECAST"], 0.10)
        if any(k in text for k in ["cause", "causal", "treatment effect", "원인", "인과", "효과"]):
            scores["CAUSAL_ANALYSIS"] = max(scores["CAUSAL_ANALYSIS"], 0.93)
            evidence.append("explicit:causal")

        if any(k in text for k in ["audit only", "analysis only", "eda only", "점검만", "분석만", "학습 없이"]):
            scores["AUDIT_DATA"] = max(scores["AUDIT_DATA"], 0.92)
            scores["ANALYZE_ONLY"] = max(scores["ANALYZE_ONLY"], 0.90)
            if "TRAIN_MODEL" not in negated:
                negated.append("TRAIN_MODEL")
            scores["TRAIN_MODEL"] = 0.0
            scores["DL_TRAIN"] = 0.0
            evidence.append("explicit:analysis_only")

        for blocked in negated:
            scores[blocked]=0.0
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        thresholds = {
            "ANALYZE_ONLY":0.17,"AUDIT_DATA":0.17,"TRAIN_MODEL":0.17,"COMPARE_MODELS":0.20,
            "FORECAST":0.35,"MONITOR_EXISTING_MODEL":0.35,"DEPLOY_MODEL":0.35,"REALTIME_PIPELINE":0.35,
            "CAUSAL_ANALYSIS":0.35,"SURVIVAL_ANALYSIS":0.35,"VISION_TRAIN":0.35,"DL_TRAIN":0.35,"ARCHITECTURE":0.35,
        }
        intents = [name for name, score in ranked if score >= thresholds.get(name,0.25)]
        if not intents:
            intents = ["ANALYZE_ONLY"]
            scores["ANALYZE_ONLY"] = max(scores["ANALYZE_ONLY"], 0.40)
        primary = intents[0]

        for strong in ["MONITOR_EXISTING_MODEL", "SURVIVAL_ANALYSIS", "FORECAST", "REALTIME_PIPELINE", "ARCHITECTURE", "VISION_TRAIN"]:
            if strong in intents and scores[strong] >= 0.80:
                primary = strong
                break
        if "TRAIN_MODEL" in intents and primary in {"ANALYZE_ONLY", "COMPARE_MODELS"}:
            primary = "TRAIN_MODEL"

        top = scores.get(primary, 0.0)
        second = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0.0
        confidence = float(min(0.99, max(0.25, 0.55 + (top - second))))
        return IntentDecision(primary, intents, scores, negated, evidence, confidence)


def _maybe_datetime_column(df: pd.DataFrame) -> str | None:
    preferred = [c for c in df.columns if any(k in c.lower() for k in ["timestamp", "datetime", "date", "time"])]
    for c in preferred:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() >= 0.80:
                return c
        except Exception:
            pass
    return None


def _group_candidate(df: pd.DataFrame, target: str | None) -> str | None:
    n = max(len(df), 1)
    candidates = []
    for c in df.columns:
        if c == target:
            continue
        nunique = int(df[c].nunique(dropna=True))
        ratio = nunique / n
        repeated = 2 <= nunique < n and ratio <= 0.80
        name=c.lower()
        semantic = any(k in name for k in ["id", "subject", "patient", "equipment", "machine", "product", "lot", "user", "device", "serial", "camera"])
        if repeated and semantic:
            if any(k in name for k in ["subject", "patient", "product", "lot", "equipment", "machine", "device", "serial", "user"]): rank=0
            elif name.endswith("_id") and "camera" not in name: rank=1
            elif "camera" in name: rank=3
            else: rank=2
            candidates.append((rank, ratio, c))
    return sorted(candidates)[0][2] if candidates else None


def infer_problem_type(profile: dict[str, Any], df: pd.DataFrame | None = None) -> str:
    explicit = str(profile.get("target_type", "")).lower()
    if profile.get("censor_col") or profile.get("event_col"):
        return "survival"
    if explicit in {"categorical", "classification", "binary", "multiclass", "class"}:
        return "classification"
    if explicit in {"continuous", "regression", "numeric", "count"}:
        return "regression"
    target = profile.get("target")
    if df is not None and target in df.columns:
        y = df[target]
        if not pd.api.types.is_numeric_dtype(y):
            return "classification"
        if y.nunique(dropna=True) <= max(20, int(len(y) * 0.02)):
            return "classification"
        return "regression"
    return "analysis"


class TaskSpecBuilder:
    def __init__(self, classifier: IntentClassifier | None = None):
        self.classifier = classifier or IntentClassifier()

    def build(self, problem: dict[str, Any]) -> dict[str, Any]:
        p = dict(problem.get("profile", {}))
        task = problem.get("task", "") or ""
        path = problem.get("data_path")
        df: pd.DataFrame | None = None
        if path and Path(path).exists() and Path(path).suffix.lower() == ".csv":
            try:
                df = pd.read_csv(path)
            except Exception:
                df = None

        intent = self.classifier.classify(task, p)
        target = p.get("target")
        problem_type = infer_problem_type(p, df)
        timestamp = p.get("timestamp_col") or (_maybe_datetime_column(df) if df is not None else None)
        group_id = p.get("group_id") or (_group_candidate(df, target) if df is not None else None)

        observation_unit = p.get("observation_unit")
        if not observation_unit:
            if group_id and timestamp:
                observation_unit = f"one row per {group_id} at {timestamp}"
            elif group_id:
                observation_unit = f"one row per observation; repeated entity={group_id}"
            else:
                observation_unit = "one row per recorded observation (domain confirmation required)"

        prediction_time = p.get("prediction_time") or "UNKNOWN"
        available_features: list[str] | str
        if df is not None:
            available_features = [str(c) for c in df.columns if c != target]
        else:
            available_features = p.get("available_features", "UNKNOWN")

        if problem_type == "classification":
            primary_metric = p.get("primary_metric") or "macro_f1 / balanced_accuracy; probability quality if decisions use probabilities"
        elif problem_type == "survival":
            primary_metric = p.get("primary_metric") or "c_index + time-dependent calibration/discrimination"
        elif intent.primary_intent == "FORECAST":
            primary_metric = p.get("primary_metric") or "MAE/RMSE by forecast horizon vs seasonal naive"
        elif problem_type == "regression":
            primary_metric = p.get("primary_metric") or "RMSE + MAE"
        else:
            primary_metric = p.get("primary_metric") or "task-specific evidence"

        if intent.primary_intent == "FORECAST" and "FORECAST" not in intent.negated:
            split_strategy = "chronological / rolling backtest"
        elif group_id:
            split_strategy = f"group-aware split by {group_id}"
        elif problem_type == "classification":
            split_strategy = "stratified holdout + train-only CV"
        elif problem_type == "regression":
            split_strategy = "holdout + train-only CV; replace with time/group split when domain metadata exists"
        else:
            split_strategy = "not applicable / analysis-only"

        causal_or_predictive = "causal" if "CAUSAL_ANALYSIS" in intent.intents else ("predictive" if target and "TRAIN_MODEL" in intent.intents else "descriptive")
        business_cost = p.get("business_cost") or p.get("cost_matrix") or "UNKNOWN"
        deployment_constraints = {
            "latency_sla": p.get("latency_sla") or p.get("sla"),
            "memory_mb": p.get("memory_mb"),
            "device": p.get("device"),
            "streaming": bool(p.get("streaming", False)),
            "size_gb": p.get("size_gb"),
        }

        unknowns = []
        if target and prediction_time == "UNKNOWN":
            unknowns.append("prediction_time")
        if observation_unit.endswith("domain confirmation required)"):
            unknowns.append("observation_unit")
        if business_cost == "UNKNOWN" and problem_type in {"classification", "regression"}:
            unknowns.append("business_cost")
        if group_id is None and df is not None and len(df) > 0:
            unknowns.append("group/entity id (if repeated entities exist)")

        return {
            "observation_unit": observation_unit,
            "target": target,
            "prediction_time": prediction_time,
            "available_features": available_features,
            "group_id": group_id,
            "timestamp": timestamp,
            "problem_type": problem_type,
            "split_strategy": split_strategy,
            "primary_metric": primary_metric,
            "business_cost": business_cost,
            "causal_or_predictive": causal_or_predictive,
            "deployment_constraints": deployment_constraints,
            "intent": asdict(intent),
            "unknowns": unknowns,
        }
