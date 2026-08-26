from __future__ import annotations
from typing import Any
import re
import numpy as np
import pandas as pd
from data_guard_v4 import analyze_dataframe as analyze_v4

GENERIC_GROUP_SUFFIX_RE = re.compile(r'(?:^|_)(?:ref|bundle|cohort|owner|party|cluster|family|crew|merchant|case|voyage|cohort|panel|fold|member|team)(?:$|_)', re.I)
GENERIC_GROUP_TOKEN_RE = re.compile(r'(crew|family|merchant|case|voyage|customer|client|account|patient|subject|person|user|device|driver|source|household|molecule|series|group|batch|lot|session|cohort|owner|party|team)', re.I)
TIME_NAME_RE = re.compile(r'timestamp|datetime|date|time', re.I)


def _target_proxy_stats(x: pd.Series, y: pd.Series) -> dict[str, Any] | None:
    mask=x.notna() & y.notna()
    if int(mask.sum()) < max(20, int(0.5*len(y))):
        return None
    xx=x[mask]; yy=y[mask]
    try:
        exact=float((xx.astype(str).to_numpy()==yy.astype(str).to_numpy()).mean())
    except Exception:
        exact=0.0
    out={'exact_match_rate': exact}
    if pd.api.types.is_numeric_dtype(xx) and pd.api.types.is_numeric_dtype(yy):
        a=pd.to_numeric(xx,errors='coerce'); b=pd.to_numeric(yy,errors='coerce')
        try:
            corr=float(a.corr(b)) if a.nunique()>1 and b.nunique()>1 else None
        except Exception:
            corr=None
        out['pearson']=corr
        if corr is not None and np.isfinite(corr):
            scale=float(np.nanstd(b.to_numpy())) or 1.0
            try:
                coef=np.polyfit(b.to_numpy(dtype=float),a.to_numpy(dtype=float),1)
                pred=coef[0]*b.to_numpy(dtype=float)+coef[1]
                nrmse=float(np.sqrt(np.mean((a.to_numpy(dtype=float)-pred)**2))/scale)
                out['affine_nrmse']=nrmse
            except Exception:
                out['affine_nrmse']=None
    return out


def analyze_dataframe(df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    guard=analyze_v4(df,target=target)
    n=max(len(df),1)
    entity=list(guard.get('entity_candidates') or [])
    entity_names={str(x.get('column')) for x in entity}

    for c in df.columns:
        if c==target or str(c) in entity_names:
            continue
        s=df[c]; nunique=int(s.nunique(dropna=True)); ratio=nunique/n
        repeated=2 <= nunique < n and ratio <= 0.65
        name=str(c); lower=name.lower()
        semantic=bool(GENERIC_GROUP_TOKEN_RE.search(lower) or GENERIC_GROUP_SUFFIX_RE.search(lower))
        suffix=any(lower.endswith(x) for x in ('_ref','_bundle','_cohort','_owner','_party','_cluster','_family','_crew','_team'))
        structural=(not pd.api.types.is_numeric_dtype(s)) and nunique>=5 and ratio<=0.50
        if repeated and (semantic or suffix or structural):
            sizes=s.astype(str).value_counts(dropna=False)
            entity.append({'column':name,'unique_ratio':ratio,'nunique':nunique,
                           'confidence':'HIGH' if semantic or suffix else 'MEDIUM',
                           'reason':'generalized_repeated_entity_semantics' if semantic or suffix else 'repeated_object_identifier'})
            entity_names.add(name)

    times=[]
    for c in df.columns:
        if not TIME_NAME_RE.search(str(c)):
            continue
        try:
            parsed=pd.to_datetime(df[c],errors='coerce')
        except Exception:
            continue
        valid=float(parsed.notna().mean())
        if valid<0.8:
            continue
        p=parsed.dropna(); deltas=p.diff().dropna()
        neg=int((deltas < pd.Timedelta(0)).sum()) if len(deltas) else 0
        dup=int(p.duplicated().sum())
        positive=deltas[deltas>pd.Timedelta(0)]
        dominant=None; irregular=0; max_ratio=None
        if len(positive):
            vc=positive.value_counts(); dominant=vc.index[0]
            tol=max(pd.Timedelta(microseconds=1), dominant*0.05)
            irregular=int(((positive-dominant).abs()>tol).sum())
            try:
                max_ratio=float(positive.max()/dominant) if dominant>pd.Timedelta(0) else None
            except Exception:
                max_ratio=None
        times.append({'column':str(c),'valid_rate':valid,'monotonic':bool(p.is_monotonic_increasing),
                      'non_monotonic_count':neg,'duplicate_count':dup,
                      'dominant_interval':str(dominant) if dominant is not None else None,
                      'irregular_interval_count':irregular,'max_gap_ratio':max_ratio})

    drops=set(guard.get('drop_feature_columns') or [])
    warnings=list(guard.get('warnings') or [])
    target_proxies=[]
    if target and target in df.columns:
        y=df[target]
        for c in df.columns:
            if c==target: continue
            stats=_target_proxy_stats(df[c],y)
            if not stats: continue
            exact=stats.get('exact_match_rate',0.0); corr=stats.get('pearson'); nrmse=stats.get('affine_nrmse')
            high=(exact>=0.995) or (corr is not None and abs(corr)>=0.9995 and nrmse is not None and nrmse<=0.01)
            if high:
                target_proxies.append({'column':str(c),**stats,'reason':'near_direct_target_copy_or_affine_proxy'}); drops.add(str(c))
        if target_proxies:
            warnings.append({'code':'TARGET_LEAKAGE_PROXY','severity':'CRITICAL','message':f"Near-direct target copies detected: {[x['column'] for x in target_proxies]}"})

    entity.sort(key=lambda x:(0 if x.get('confidence')=='HIGH' else 1, x.get('unique_ratio',1.0)))
    guard['entity_candidates']=entity
    if entity and not guard.get('group_strategy'):
        guard['group_strategy']={'type':'column','column':entity[0]['column'],'reason':'generalized_repeated_entity_identifier','confidence':entity[0].get('confidence','MEDIUM')}
    guard['time_candidates']=times
    guard['target_leakage_proxies']=target_proxies
    guard['drop_feature_columns']=sorted(drops)
    guard['safe_feature_columns']=[c for c in df.columns if c!=target and str(c) not in drops]
    guard['warnings']=warnings
    return guard
