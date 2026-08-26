from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import json
import math
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
try:
    import faiss
except Exception:
    faiss = None

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[가-힣]{2,}")

def _tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text or "")]

def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0: return values
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12: return np.zeros_like(values) if hi <= 0 else np.ones_like(values)
    return (values - lo) / (hi - lo)

class HybridDomainRAG:
    def __init__(self, default_root: str | Path | None = None, embedding_model: str | None = None):
        self.default_root = Path(default_root) if default_root else None
        self.embedding_model = embedding_model

    @staticmethod
    def _chunk_text(text: str, source: str, max_chars: int = 1400) -> list[dict[str, Any]]:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks=[]; buf=""; heading=None; idx=0
        for p in paras:
            if p.startswith("#"): heading = p.lstrip("# ").strip()
            if buf and len(buf)+len(p)+2>max_chars:
                chunks.append({"source":source,"chunk_id":idx,"text":buf,"heading":heading}); idx+=1; buf=p
            else: buf=f"{buf}\n\n{p}".strip()
        if buf: chunks.append({"source":source,"chunk_id":idx,"text":buf,"heading":heading})
        return chunks

    def _collect_paths(self, problem):
        p=problem.get("profile",{}); paths=[]
        if self.default_root and self.default_root.exists(): paths.extend(x for x in self.default_root.rglob("*") if x.is_file() and x.suffix.lower() in {".md",".txt",".json"})
        for raw in p.get("domain_paths",[]) or []:
            q=Path(raw)
            if q.is_dir(): paths.extend(x for x in q.rglob("*") if x.is_file() and x.suffix.lower() in {".md",".txt",".json"})
            elif q.is_file(): paths.append(q)
        seen=set(); out=[]
        for x in paths:
            key=str(x.resolve())
            if key not in seen: seen.add(key); out.append(x)
        return out

    def _load(self, problem):
        chunks=[]; facts=[]; errors=[]
        for path in self._collect_paths(problem):
            try:
                if path.suffix.lower()==".json":
                    obj=json.loads(path.read_text(encoding="utf-8"))
                    for fact in (obj.get("facts",[]) if isinstance(obj,dict) else []):
                        if isinstance(fact,dict): f=dict(fact); f.setdefault("source",str(path)); facts.append(f)
                    text=json.dumps(obj,ensure_ascii=False,indent=2)
                else: text=path.read_text(encoding="utf-8")
                for chunk in self._chunk_text(text,str(path)):
                    chunk["source_type"]="structured" if path.suffix.lower()==".json" else "document"; chunks.append(chunk)
            except Exception as exc: errors.append({"source":str(path),"error":f"{type(exc).__name__}: {exc}"})
        return chunks,facts,errors

    @staticmethod
    def _bm25(corpus, query, k1=1.5, b=.75):
        docs=[_tokens(x) for x in corpus]; q=_tokens(query); N=len(docs)
        if N==0 or not q:return np.zeros(N)
        avgdl=sum(len(d) for d in docs)/max(N,1); df=Counter()
        for d in docs:
            for t in set(d):df[t]+=1
        idf={t:math.log(1+(N-df.get(t,0)+.5)/(df.get(t,0)+.5)) for t in set(q)}; scores=np.zeros(N)
        for i,d in enumerate(docs):
            tf=Counter(d); dl=len(d)
            for t in q:
                f=tf.get(t,0)
                if not f:continue
                denom=f+k1*(1-b+b*dl/max(avgdl,1e-9)); scores[i]+=idf.get(t,0)*f*(k1+1)/denom
        return scores

    def _vector_scores(self, corpus, query, profile):
        model_name=profile.get("embedding_model") or self.embedding_model
        if SentenceTransformer is not None and model_name:
            try:
                model=SentenceTransformer(model_name,local_files_only=True); emb=model.encode(corpus+[query],normalize_embeddings=True,show_progress_bar=False); docs=np.asarray(emb[:-1],dtype="float32"); q=np.asarray(emb[-1:],dtype="float32")
                if faiss is not None:
                    index=faiss.IndexFlatIP(docs.shape[1]);index.add(docs);sims,ids=index.search(q,len(corpus));out=np.zeros(len(corpus))
                    for score,idx in zip(sims[0],ids[0]):out[int(idx)]=float(score)
                    return out,f"sentence-transformers+faiss:{model_name}"
                return docs@q[0],f"sentence-transformers+numpy:{model_name}"
            except Exception:pass
        vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),lowercase=True,sublinear_tf=True);mat=vec.fit_transform(corpus+[query]);return cosine_similarity(mat[-1],mat[:-1])[0],"char-tfidf-vector-fallback"

    @staticmethod
    def _metadata_boost(chunk, problem, query):
        p=problem.get("profile",{});boost=0.;source=(chunk.get("source") or "").lower();text=(chunk.get("text") or "").lower()
        for key in [str(p.get("domain","")),str(p.get("process","")),str(p.get("target",""))]:
            key=key.strip().lower()
            if key and (key in source or key in text):boost+=.05
        if any(k in source for k in ["spec","standard","dictionary","constraint","quality"]):boost+=.04
        return min(boost,.20)

    def retrieve(self, problem, top_k=6):
        chunks,facts,errors=self._load(problem)
        if not chunks:return {"status":"NO_CONTEXT","matches":[],"facts":[],"errors":errors,"retrieval_backend":"none"}
        p=problem.get("profile",{});query=" ".join([str(problem.get("task","")),str(p.get("target","")),str(p.get("modality","")),str(p.get("domain","")),str(p.get("process",""))]).strip();corpus=[c["text"] for c in chunks];bm25=self._bm25(corpus,query);vec,backend=self._vector_scores(corpus,query,p);bmn=_minmax(bm25);vecn=_minmax(vec);combined=.45*bmn+.50*vecn+np.array([self._metadata_boost(c,problem,query) for c in chunks]);order=combined.argsort()[::-1];matches=[]
        for rank,i in enumerate(order[:top_k],1):
            score=float(combined[i])
            if score<.08:continue
            item=dict(chunks[int(i)]);item.update({"score":score,"bm25_score":float(bmn[i]),"vector_score":float(vecn[i]),"rank":rank,"retrieval_backend":backend});matches.append(item)
        hit_sources={m["source"] for m in matches};selected_facts=[f for f in facts if f.get("source") in hit_sources];return {"status":"FOUND" if matches else "NO_MATCH","matches":matches,"facts":selected_facts,"errors":errors,"retrieval_backend":backend,"query":query}
