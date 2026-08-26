from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any

@dataclass(frozen=True)
class CompetitionSpec:
    slug: str; name: str; category: str; target: str; metric: str; direction: str
    validation: str; submission_columns: tuple[str,...]; risk_flags: tuple[str,...]=(); classes: int=2
    source_url: str=''; data_mode: str='PROXY_DATA'
    def to_dict(self): return asdict(self)

class CompetitionSpecBuilderV6:
    METRIC_ALIASES={
        'rmsle':'root mean squared log error','rmse':'root mean squared error','mae':'mean absolute error',
        'r2':'r-squared','accuracy':'accuracy','roc_auc':'roc auc','log_loss':'log loss','gini':'normalized gini',
        'smape':'symmetric mean absolute percentage error','rmspe':'root mean squared percentage error',
        'weighted_mae':'weighted mean absolute error','nwrmsle':'normalized weighted rmsle','wrmsse':'weighted rmsse',
        'pinball':'pinball loss','quadratic_kappa':'quadratic weighted kappa','roc_auc_macro':'macro roc auc'
    }
    def build(self, raw:dict[str,Any])->CompetitionSpec:
        return CompetitionSpec(
            slug=raw['slug'],name=raw['name'],category=raw['category'],target=raw['target'],metric=raw['metric'],
            direction=raw['direction'],validation=raw['validation'],submission_columns=tuple(raw.get('submission_columns',[])),
            risk_flags=tuple(raw.get('risk_flags',[])),classes=int(raw.get('classes',2)),source_url=raw.get('source_url',''),
            data_mode=raw.get('data_mode','PROXY_DATA'))
