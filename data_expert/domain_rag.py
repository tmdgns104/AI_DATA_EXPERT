from __future__ import annotations
from pathlib import Path
import json,re
from typing import Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DomainRAG:
    def __init__(self,default_root=None):self.default_root=Path(default_root) if default_root else None
    @staticmethod
    def _chunk_text(text,source,max_chars=1200):
        paras=[p.strip() for p in re.split(r"\n\s*\n",text) if p.strip()];chunks=[];buf="";idx=0
        for p in paras:
            if buf and len(buf)+len(p)+2>max_chars:chunks.append({"source":source,"chunk_id":idx,"text":buf});idx+=1;buf=p
            else:buf=f"{buf}\n\n{p}".strip()
        if buf:chunks.append({"source":source,"chunk_id":idx,"text":buf})
        return chunks
    def _collect_paths(self,problem):
        p=problem.get("profile",{});paths=[]
        if self.default_root and self.default_root.exists():paths += [x for x in self.default_root.rglob("*") if x.is_file() and x.suffix.lower() in {".md",".txt",".json"}]
        for raw in p.get("domain_paths",[]) or []:
            q=Path(raw)
            if q.is_dir():paths += [x for x in q.rglob("*") if x.is_file() and x.suffix.lower() in {".md",".txt",".json"}]
            elif q.is_file():paths.append(q)
        seen=set();out=[]
        for x in paths:
            key=str(x.resolve())
            if key not in seen:seen.add(key);out.append(x)
        return out
    def retrieve(self,problem,top_k=4):
        chunks=[];errors=[]
        for path in self._collect_paths(problem):
            try:
                text=json.dumps(json.loads(path.read_text(encoding="utf-8")),ensure_ascii=False,indent=2) if path.suffix.lower()==".json" else path.read_text(encoding="utf-8");chunks.extend(self._chunk_text(text,str(path)))
            except Exception as exc:errors.append({"source":str(path),"error":f"{type(exc).__name__}: {exc}"})
        if not chunks:return {"status":"NO_CONTEXT","matches":[],"errors":errors}
        p=problem.get("profile",{});query=" ".join([problem.get("task",""),str(p.get("target","")),str(p.get("modality","")),str(p.get("domain","")),str(p.get("process",""))]).strip();corpus=[c["text"] for c in chunks];vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),lowercase=True,sublinear_tf=True);matrix=vec.fit_transform(corpus+[query]);sims=cosine_similarity(matrix[-1],matrix[:-1])[0];order=sims.argsort()[::-1];matches=[]
        for i in order[:top_k]:
            score=float(sims[i])
            if score<=.08:continue
            item=dict(chunks[int(i)]);item["score"]=score;matches.append(item)
        return {"status":"FOUND" if matches else "NO_MATCH","matches":matches,"errors":errors}
