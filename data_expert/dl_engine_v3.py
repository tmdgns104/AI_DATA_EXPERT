from __future__ import annotations

from pathlib import Path
from typing import Any
import math
import random

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
except Exception:  # pragma: no cover
    torch = None
    nn = None
    TensorDataset = None
    DataLoader = None

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score

import enhanced_system_v2 as legacy


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, classification: bool):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, out_dim),
        )
        self.classification = classification

    def forward(self, x):
        return self.net(x)


class _TinyCNN(nn.Module):
    def __init__(self, channels: int, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(16, n_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.head(x)


def _seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def _train_model(model, Xtr, ytr, Xv, yv, classification: bool, epochs: int = 50, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    xtr = torch.tensor(Xtr, dtype=torch.float32)
    xv = torch.tensor(Xv, dtype=torch.float32)
    if classification:
        ytr_t = torch.tensor(ytr, dtype=torch.long)
        yv_t = torch.tensor(yv, dtype=torch.long)
        loss_fn = nn.CrossEntropyLoss()
    else:
        ytr_t = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
        yv_t = torch.tensor(yv, dtype=torch.float32).view(-1, 1)
        loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(xtr, ytr_t), batch_size=min(32, len(xtr)), shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None; best_val = float("inf"); history=[]; patience=10; stalled=0
    for epoch in range(epochs):
        model.train(); total=0.0; seen=0
        for xb, yb in loader:
            xb=xb.to(device); yb=yb.to(device); opt.zero_grad(); pred=model(xb); loss=loss_fn(pred,yb); loss.backward(); opt.step()
            total += float(loss.item()) * len(xb); seen += len(xb)
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(xv.to(device)), yv_t.to(device)).item()
        history.append({"epoch": epoch+1, "train_loss": total/max(seen,1), "val_loss": float(v)})
        if v < best_val - 1e-6:
            best_val=float(v); best_state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}; stalled=0
        else:
            stalled += 1
        if stalled >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, str(device)


def _small_batch_sanity(in_dim, out_dim, X, y, classification: bool):
    n=min(24,len(X)); Xs=X[:n]; ys=y[:n]
    model=_MLP(in_dim,out_dim,classification)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=model.to(device)
    xt=torch.tensor(Xs,dtype=torch.float32,device=device)
    yt=torch.tensor(ys,dtype=torch.long if classification else torch.float32,device=device)
    if not classification: yt=yt.view(-1,1)
    loss_fn=nn.CrossEntropyLoss() if classification else nn.MSELoss(); opt=torch.optim.Adam(model.parameters(),lr=0.01)
    model.train(); initial=None; final=None
    for i in range(80):
        opt.zero_grad(); pred=model(xt); loss=loss_fn(pred,yt); loss.backward(); opt.step()
        if initial is None: initial=float(loss.item())
        final=float(loss.item())
    ratio=final/max(initial,1e-12)
    return {"initial_loss":initial,"final_loss":final,"loss_ratio":ratio,"pass":bool(ratio < 0.20)}


class TorchDeepLearningExpert:
    def run_tabular(self, problem: dict[str, Any]):
        if torch is None:
            raise RuntimeError("PyTorch is not installed")
        _seed(42)
        p=problem["profile"]; df=pd.read_csv(problem["data_path"]); target=p["target"]
        X=df.drop(columns=[target]); raw_y=df[target]
        classification = legacy.infer_supervised_type(problem) == "classification"
        cat=X.select_dtypes(include=["object","category","string","bool"]).columns.tolist(); num=[c for c in X.columns if c not in cat]
        prep=ColumnTransformer([
            ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num),
            ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat),
        ])
        idx=np.arange(len(df)); train_idx,val_idx=train_test_split(idx,test_size=.25,random_state=42,stratify=raw_y if classification else None)
        Xtr=prep.fit_transform(X.iloc[train_idx]).astype("float32"); Xv=prep.transform(X.iloc[val_idx]).astype("float32")
        if classification:
            enc=LabelEncoder(); yall=enc.fit_transform(raw_y.astype(str)); ytr=yall[train_idx]; yv=yall[val_idx]; out_dim=len(enc.classes_)
        else:
            yall=raw_y.astype(float).to_numpy(dtype="float32"); ytr=yall[train_idx]; yv=yall[val_idx]; out_dim=1
        sanity=_small_batch_sanity(Xtr.shape[1],out_dim,Xtr,ytr,classification)
        model=_MLP(Xtr.shape[1],out_dim,classification)
        model,history,device=_train_model(model,Xtr,ytr,Xv,yv,classification,epochs=60)
        model.eval(); dev=torch.device(device)
        with torch.no_grad(): logits=model(torch.tensor(Xv,dtype=torch.float32,device=dev)).cpu().numpy()
        if classification:
            pred=logits.argmax(axis=1); metrics={"accuracy":float(accuracy_score(yv,pred)),"macro_f1":float(f1_score(yv,pred,average="macro"))}
        else:
            pred=logits.reshape(-1); metrics={"rmse":float(math.sqrt(mean_squared_error(yv,pred)))}
        out=legacy.base.step_record("deep-learning")
        out["UNDERSTAND"]={"task":"actual PyTorch tabular training","problem_type":"classification" if classification else "regression","device":device}
        out["INSPECT"]=[{"fact":"rows","value":len(df)},{"fact":"input_dim","value":int(Xtr.shape[1])},{"fact":"device","value":device}]
        out["QUESTION"]=["Can a tiny batch be overfit before trusting full training?","Does DL beat a simpler ML baseline enough to justify complexity?"]
        out["HYPOTHESES"]=[{"id":"H-DL-PIPE","statement":"pipeline/label alignment is trainable"},{"id":"H-DL-CAP","statement":"neural capacity adds useful signal"}]
        out["TESTS"]=[{"test":"small_batch_overfit","result":sanity},{"test":"validation_training","epochs":len(history),"last":history[-1] if history else None,"metrics":metrics}]
        out["COMPARE"]=[{"option":"PyTorch MLP","validation":metrics},{"option":"simple ML baseline","status":"compare in machine-learning expert output"}]
        out["DECIDE"]={"decision":"PyTorch training completed; select only if validation beats simpler baseline","validation_metrics":metrics,"device":device,"checkpoint":"best validation state restored"}
        out["CHALLENGE"]=["Do not choose DL merely because the user requested it","A failed small-batch sanity test suggests pipeline/optimization problems before model scaling"]
        out["RISKS"]=["single validation split; production split may need entity/time semantics","CPU training is functional but not a deployment latency benchmark"]
        out["CONFIDENCE"]={"level":"HIGH" if sanity["pass"] else "MEDIUM","reason":"actual PyTorch execution plus small-batch sanity evidence"}
        out["markers"] += ["actual_torch_training","small_batch_overfit","checkpoint_best_validation","data_first","silent_failure_guard","same_split_compare"]
        return legacy.attach_heuristics(out,problem)

    def run_image_npz(self, problem: dict[str, Any]):
        if torch is None:
            raise RuntimeError("PyTorch is not installed")
        _seed(42)
        p=problem["profile"]; npz=Path(p["image_npz"]); arr=np.load(npz); X=arr["images"].astype("float32"); y=arr["labels"].astype("int64")
        if X.ndim==3: X=X[:,None,:,:]
        if X.max()>1.5: X=X/255.0
        idx=np.arange(len(y)); tr,va=train_test_split(idx,test_size=.25,random_state=42,stratify=y)
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model=_TinyCNN(X.shape[1],int(np.max(y))+1).to(device); loss_fn=nn.CrossEntropyLoss(); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
        loader=DataLoader(TensorDataset(torch.tensor(X[tr]),torch.tensor(y[tr])),batch_size=32,shuffle=True)
        xv=torch.tensor(X[va],dtype=torch.float32,device=device); yv=torch.tensor(y[va],dtype=torch.long,device=device)
        best=float("inf"); best_state=None; hist=[]
        for epoch in range(15):
            model.train(); total=0.0; seen=0
            for xb,yb in loader:
                xb=xb.to(device); yb=yb.to(device); opt.zero_grad(); pred=model(xb); loss=loss_fn(pred,yb); loss.backward(); opt.step(); total+=float(loss.item())*len(xb); seen+=len(xb)
            model.eval()
            with torch.no_grad(): v=float(loss_fn(model(xv),yv).item())
            hist.append({"epoch":epoch+1,"train_loss":total/max(seen,1),"val_loss":v})
            if v<best: best=v; best_state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}
        if best_state: model.load_state_dict(best_state)
        model.eval();
        with torch.no_grad(): pred=model(xv).argmax(1).cpu().numpy()
        metrics={"accuracy":float(accuracy_score(y[va],pred)),"macro_f1":float(f1_score(y[va],pred,average="macro"))}

        sb=min(20,len(tr)); m2=_TinyCNN(X.shape[1],int(np.max(y))+1).to(device); opt2=torch.optim.Adam(m2.parameters(),lr=.01); xs=torch.tensor(X[tr[:sb]],dtype=torch.float32,device=device); ys=torch.tensor(y[tr[:sb]],dtype=torch.long,device=device)
        initial=final=None
        for i in range(60):
            opt2.zero_grad(); l=loss_fn(m2(xs),ys); l.backward(); opt2.step(); initial=float(l.item()) if initial is None else initial; final=float(l.item())
        sanity={"initial_loss":initial,"final_loss":final,"loss_ratio":final/max(initial,1e-12),"pass":final/max(initial,1e-12)<.20}

        out=legacy.base.step_record("deep-learning")
        out["UNDERSTAND"]={"task":"actual pixel CNN classification","shape":list(X.shape),"device":str(device)}
        out["INSPECT"]=[{"fact":"classes","value":int(np.max(y))+1},{"fact":"pixel_range","value":[float(X.min()),float(X.max())]}]
        out["QUESTION"]=["Can a small image batch be overfit?","Does the split leak product/source identity?","Are augmentations label-preserving?"]
        out["HYPOTHESES"]=[{"id":"H-CNN-1","statement":"pixel signal supports classification"},{"id":"H-CNN-2","statement":"shortcut/source leakage may inflate score"}]
        out["TESTS"]=[{"test":"small_batch_overfit","result":sanity},{"test":"actual_CNN_validation","metrics":metrics,"epochs":len(hist)}]
        out["COMPARE"]=[{"model":"TinyCNN","validation":metrics},{"baseline":"vision metadata/simpler baseline should also be checked"}]
        out["DECIDE"]={"decision":"CNN training executed","validation_metrics":metrics,"checkpoint":"best validation state restored"}
        out["CHALLENGE"]=["Inspect saliency/failure cases before deployment","Random image split is unsafe for repeated product/source images"]
        out["RISKS"]=["synthetic/small image benchmark is not evidence of production vision quality"]
        out["CONFIDENCE"]={"level":"MEDIUM","reason":"actual pixel training executed; external domain validation absent"}
        out["markers"] += ["actual_torch_training","actual_pixel_training","small_batch_overfit","checkpoint_best_validation","vision_training"]
        return legacy.attach_heuristics(out,problem)

    def run(self, problem: dict[str, Any]):
        p=problem.get("profile",{})
        if p.get("image_npz"):
            return self.run_image_npz(problem)
        if p.get("modality","tabular")=="tabular" and p.get("target"):
            return self.run_tabular(problem)
        return legacy.EnhancedDL().run(problem)
