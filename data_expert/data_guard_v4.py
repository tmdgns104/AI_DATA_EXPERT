from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import re

import numpy as np
import pandas as pd


ID_NAME_RE = re.compile(r"(^|_)(id|uuid|guid|index|row|record|serial|key|seq_no|sequence_id)($|_)", re.I)
ENTITY_NAME_RE = re.compile(r"(^|_)(patient|subject|user|device|equipment|machine|product|lot|batch|wafer|vehicle|camera|mold|tool|station|unit)(_|$)", re.I)
SEQUENCE_NAME_RE = re.compile(r"(^|_)(shot|cycle|sequence|seq|step|run_no|counter|count)($|_)", re.I)
TIME_NAME_RE = re.compile(r"timestamp|datetime|date|time", re.I)


def _safe_float(v: Any) -> float | None:
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def _spearman_with_row(s: pd.Series) -> float | None:
    if not pd.api.types.is_numeric_dtype(s):
        return None
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() < max(10, int(len(x) * 0.8)) or x.nunique(dropna=True) < 5:
        return None
    row = pd.Series(np.arange(len(x)), index=x.index)
    try:
        c = x.corr(row, method="spearman")
        return _safe_float(c)
    except Exception:
        return None


def derive_reset_groups(series: pd.Series) -> tuple[pd.Series | None, dict[str, Any] | None]:
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().mean() < 0.9 or x.nunique(dropna=True) < 5:
        return None, None
    diffs = x.diff()
    span = float(x.max() - x.min()) if x.notna().any() else 0.0
    resets = diffs < (-max(span * 0.15, 1.0))
    reset_count = int(resets.fillna(False).sum())
    if reset_count < 1:
        return None, None
    groups = resets.fillna(False).cumsum().astype(int)
    sizes = groups.value_counts().sort_index()
    if len(sizes) < 2 or int(sizes.min()) < 3:
        return None, None
    return groups, {
        "reset_count": reset_count,
        "n_groups": int(groups.nunique()),
        "min_group_size": int(sizes.min()),
        "max_group_size": int(sizes.max()),
    }


def analyze_dataframe(df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    n = max(len(df), 1)
    target_missing = int(df[target].isna().sum()) if target and target in df.columns else 0
    target_missing_rate = target_missing / n

    identifiers: list[dict[str, Any]] = []
    entity_candidates: list[dict[str, Any]] = []
    time_candidates: list[dict[str, Any]] = []
    reset_candidates: list[dict[str, Any]] = []
    drop_features: list[str] = []
    warnings: list[dict[str, Any]] = []

    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        nunique = int(s.nunique(dropna=True))
        ratio = nunique / n
        name = str(c)
        lower = name.lower()
        row_corr = _spearman_with_row(s)

        id_semantic = bool(ID_NAME_RE.search(lower)) or lower in {"_id", "id", "row_id", "record_id"}
        near_unique = ratio >= 0.90
        row_proxy = row_corr is not None and abs(row_corr) >= 0.97 and ratio >= 0.80
        if (id_semantic and near_unique) or row_proxy:
            reason = []
            if id_semantic and near_unique:
                reason.append("semantic_high_cardinality_identifier")
            if row_proxy:
                reason.append("near_monotonic_row_order_proxy")
            identifiers.append({"column": name, "unique_ratio": ratio, "row_spearman": row_corr, "reason": reason})
            drop_features.append(name)

        repeated = 2 <= nunique < n and ratio <= 0.80
        entity_token = bool(ENTITY_NAME_RE.search(lower))
        entity_suffix = any(lower.endswith(suf) for suf in ["_id","_code","_no","_number","_serial"])
        exact_entity = lower in {"lot","batch","wafer","device","equipment","machine","product","patient","subject","vehicle","mold"}
        if repeated and entity_token and (entity_suffix or exact_entity):
            entity_candidates.append({"column": name, "unique_ratio": ratio, "nunique": nunique})

        if TIME_NAME_RE.search(lower):
            try:
                parsed = pd.to_datetime(s, errors="coerce")
                valid = float(parsed.notna().mean())
                if valid >= 0.8:
                    time_candidates.append({"column": name, "valid_rate": valid, "monotonic": bool(parsed.is_monotonic_increasing)})
            except Exception:
                pass

        if SEQUENCE_NAME_RE.search(lower) or lower in {"shot", "cycle", "sequence"}:
            groups, info = derive_reset_groups(s)
            if groups is not None and info is not None:
                reset_candidates.append({"column": name, **info})

    def entity_rank(item: dict[str, Any]):
        name = item["column"].lower()
        if any(k in name for k in ["patient", "subject", "device", "equipment", "machine", "product", "lot", "batch", "wafer", "vehicle", "mold"]):
            p = 0
        elif "camera" in name or "source" in name:
            p = 3
        else:
            p = 1
        return (p, item["unique_ratio"])

    entity_candidates.sort(key=entity_rank)
    group_strategy: dict[str, Any] | None = None
    if entity_candidates:
        group_strategy = {"type": "column", "column": entity_candidates[0]["column"], "reason": "repeated_entity_identifier"}
    elif reset_candidates:
        best = sorted(reset_candidates, key=lambda x: (-x["reset_count"], x["column"]))[0]
        group_strategy = {"type": "derived_reset_run", "column": best["column"], "reason": "sequence_reset_implies_run_boundary", **{k: v for k, v in best.items() if k != "column"}}

    if target_missing:
        warnings.append({"code": "TARGET_MISSING", "severity": "WARNING", "message": f"{target_missing} target rows are unlabeled and must not become a class."})
    if identifiers:
        warnings.append({"code": "IDENTIFIER_PROXY", "severity": "WARNING", "message": f"Identifier/order proxy features detected: {[x['column'] for x in identifiers]}"})
    if group_strategy:
        warnings.append({"code": "GROUP_SPLIT_CANDIDATE", "severity": "INFO", "message": f"Use group-aware validation: {group_strategy}"})

    safe_features = [c for c in df.columns if c != target and c not in set(drop_features)]
    return {
        "rows": int(len(df)),
        "target": target,
        "target_missing_count": target_missing,
        "target_missing_rate": target_missing_rate,
        "labeled_count": int(len(df) - target_missing) if target and target in df.columns else None,
        "unlabeled_count": target_missing if target and target in df.columns else None,
        "identifier_proxies": identifiers,
        "drop_feature_columns": sorted(set(drop_features)),
        "safe_feature_columns": safe_features,
        "entity_candidates": entity_candidates,
        "time_candidates": time_candidates,
        "sequence_reset_candidates": reset_candidates,
        "group_strategy": group_strategy,
        "warnings": warnings,
    }


def group_series(df: pd.DataFrame, guard: dict[str, Any]) -> pd.Series | None:
    strategy = guard.get("group_strategy") or {}
    if strategy.get("type") == "column" and strategy.get("column") in df.columns:
        return df[strategy["column"]].astype(str)
    if strategy.get("type") == "derived_reset_run" and strategy.get("column") in df.columns:
        groups, _ = derive_reset_groups(df[strategy["column"]])
        return groups
    return None


def split_labeled_unlabeled(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if target not in df.columns:
        raise KeyError(f"Target {target!r} not found")
    mask = df[target].notna()
    return df.loc[mask].copy(), df.loc[~mask].copy()
