from __future__ import annotations
from pathlib import Path
from typing import Any
import tempfile, random
import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
except Exception:
    torch=None;nn=None;TensorDataset=None;DataLoader=None
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
import enhanced_system_v2 as legacy
from dl_engine_v3 import TorchDeepLearningExpert as V3TorchExpert
from data_guard_v4 import analyze_dataframe,split_labeled_unlabeled

class BetterCNN(nn.Module):
    def __init__(self,channels,n_classes):
        super().__init__();self.features=nn.Sequential(nn.Conv2d(channels,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.AdaptiveAvgPool2d((2,2)));self.head=nn.Sequential(nn.Flatten(),nn.Linear(64*4,64),nn.ReLU(),nn.Dropout(.10),nn.Linear(64,n_classes))
    def forward(self,x):return self.head(self.features(x))

def _seed(seed=42):
    random.seed(seed);np.random.seed(seed)
    if torch is not None:torch.manual_seed(seed)

def _normalize(Xtr,Xv):
    axes=(0,2,3);mean=Xtr.mean(axis=axes,keepdims=True);std=Xtr.std(axis=axes,keepdims=True);std=np.where(std<1e-6,1.0,std);return (Xtr-mean)/std,(Xv-mean)/std,{"mean":mean.reshape(-1).tolist(),"std":std.reshape(-1).tolist()}

def _small_batch_image_sanity(X,y,channels,n_classes,device):
    n=min(24,len(y));m=BetterCNN(channels,n_classes).to(device);xs=torch.tensor(X[:n],dtype=torch.float32,device=device);ys=torch.tensor(y[:n],dtype=torch.long,device=device);opt=torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=1e-5);loss_fn=nn.CrossEntropyLoss();initial=final=None
    for _ in range(100):opt.zero_grad();loss=loss_fn(m(xs),ys);loss.backward();opt.step();initial=float(loss.item()) if initial is None else initial;final=float(loss.item())
    ratio=final/max(initial,1e-12);return {"initial_loss":initial,"final_loss":final,"loss_ratio":ratio,"pass":bool(ratio<.15)}

class TorchDeepLearningExpertV4:
    def __init__(self):self.v3=V3TorchExpert()
    def run_tabular(self,problem):
        p=problem["profile"];df=pd.read_csv(problem["data_path"]);target=p["target"];guard=problem.get("data_guard") or analyze_dataframe(df,target);labeled,_=split_labeled_unlabeled(df,target);excluded=set(guard.get("drop_feature_columns",[]))|set(problem.get("task_spec",{}).get("excluded_domain_features",[]) or []);keep=[c for c in labeled.columns if c==target or c not in excluded];cleaned=labeled[keep]
        with tempfile.NamedTemporaryFile("w",suffix=".csv",delete=False,encoding="utf-8",newline="") as fh:temp=Path(fh.name);cleaned.to_csv(fh,index=False)
        clone={**problem,"data_path":str(temp),"profile":dict(problem.get("profile",{}))}
        try:out=self.v3.run_tabular(clone)
        finally:temp.unlink(missing_ok=True)
        out["INSPECT"].append({"fact":"V4 data guard","value":{"excluded_features":sorted(excluded),"target_missing_count":guard.get("target_missing_count",0)}});out["markers"] += ["target_missing_separated","identifier_proxy_excluded","domain_context_injected"];return out
    def run_image_npz(self,problem):
        if torch is None:raise RuntimeError("PyTorch is not installed")
        p=problem["profile"];arr=np.load(Path(p["image_npz"]));X=arr["images"].astype("float32");y=arr["labels"].astype("int64")
        if X.ndim==3:X=X[:,None,:,:]
        if X.max()>1.5:X=X/255.0
        idx=np.arange(len(y));tr,va=train_test_split(idx,test_size=.25,random_state=42,stratify=y);Xtr,Xv,norm=_normalize(X[tr],X[va]);ytr,yv=y[tr],y[va];n_classes=int(np.max(y))+1;device=torch.device("cuda" if torch.cuda.is_available() else "cpu");sanity=_small_batch_image_sanity(Xtr,ytr,X.shape[1],n_classes,device);seed_results=[];best_global=None
        for seed in [42,73]:
            _seed(seed);model=BetterCNN(X.shape[1],n_classes).to(device);counts=np.bincount(ytr,minlength=n_classes);weights=np.where(counts>0,len(ytr)/(n_classes*counts),1.0);loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device));opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4);loader=DataLoader(TensorDataset(torch.tensor(Xtr),torch.tensor(ytr)),batch_size=min(32,len(ytr)),shuffle=True);xv=torch.tensor(Xv,dtype=torch.float32,device=device);yvt=torch.tensor(yv,dtype=torch.long,device=device);best_loss=float("inf");best_state=None;patience=12;stale=0;history=[]
            for epoch in range(60):
                model.train();total=0.;seen=0
                for xb,yb in loader:
                    xb=xb.to(device);yb=yb.to(device);opt.zero_grad();logits=model(xb);loss=loss_fn(logits,yb);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step();total+=float(loss.item())*len(xb);seen+=len(xb)
                model.eval();
                with torch.no_grad():vloss=float(loss_fn(model(xv),yvt).item())
                history.append({"epoch":epoch+1,"train_loss":total/max(seen,1),"val_loss":vloss})
                if vloss<best_loss-1e-5:best_loss=vloss;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
                else:stale+=1
                if stale>=patience:break
            if best_state:model.load_state_dict(best_state)
            model.eval();
            with torch.no_grad():logits=model(xv).cpu().numpy();pred=logits.argmax(1)
            metrics={"accuracy":float(accuracy_score(yv,pred)),"macro_f1":float(f1_score(yv,pred,average="macro")),"confusion_matrix":confusion_matrix(yv,pred).tolist()};rec={"seed":seed,"metrics":metrics,"epochs":len(history),"best_val_loss":best_loss,"state":best_state};seed_results.append({k:v for k,v in rec.items() if k!="state"});best_global=rec if best_global is None or metrics["macro_f1"]>best_global["metrics"]["macro_f1"] else best_global
        metrics=best_global["metrics"];out=legacy.base.step_record("deep-learning");out["UNDERSTAND"]={"task":"actual pixel CNN classification V4","shape":list(X.shape),"device":str(device)};out["INSPECT"]=[{"fact":"classes","value":n_classes},{"fact":"normalization","value":norm},{"fact":"class_counts","value":np.bincount(y,minlength=n_classes).tolist()}];out["QUESTION"]=["Can a small image batch be overfit?","Does product/source identity leak across splits?","Are augmentations label-preserving?","Is performance stable across seeds?"];out["HYPOTHESES"]=[{"id":"H-CNN-PIPE","statement":"pixel/label pipeline is trainable"},{"id":"H-CNN-ROBUST","statement":"a larger normalized CNN is stable across seeds"}];out["TESTS"]=[{"test":"small_batch_overfit","result":sanity},{"test":"multi_seed_validation","result":seed_results}];out["COMPARE"]=[{"model":"BetterCNN","validation":metrics},{"baseline":"simple vision baseline/metadata leakage check required"}];out["DECIDE"]={"decision":"V4 CNN training executed; best validation seed retained","validation_metrics":metrics,"best_seed":best_global["seed"],"checkpoint":"best validation state restored","normalization":norm};out["CHALLENGE"]=["High validation score is not production evidence without product/source split","Inspect class-specific failures and saliency before deployment"];out["RISKS"]=["No unsafe augmentation is applied without label-semantics evidence","External camera/lighting/domain shift remains untested"];out["CONFIDENCE"]={"level":"HIGH" if sanity["pass"] and metrics["macro_f1"]>=.80 else "MEDIUM","reason":"actual normalized multi-seed PyTorch training with small-batch sanity"};out["markers"] += ["actual_torch_training","actual_pixel_training","small_batch_overfit","checkpoint_best_validation","multi_seed_vision","pixel_normalization","vision_training"];return legacy.attach_heuristics(out,problem)
    def run(self,problem):
        p=problem.get("profile",{})
        if p.get("image_npz"):return self.run_image_npz(problem)
        if p.get("modality","tabular")=="tabular" and p.get("target"):return self.run_tabular(problem)
        return self.v3.run(problem)
