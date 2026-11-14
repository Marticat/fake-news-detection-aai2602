"""Duplicate-family isolation and reproducible uncertainty estimates."""
import hashlib
import numpy as np
from sklearn.metrics import f1_score

def duplicate_groups(frame):
    """Conservative families: identical normalized title or 80-word body prefix.

    Short generic titles are ignored. This is a heuristic, not exhaustive
    semantic deduplication; all detected family members stay in one partition.
    """
    from .data import clean_text
    parent=list(range(len(frame)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    seen={}
    for i,row in enumerate(frame.itertuples()):
        title=clean_text(row.title)
        words=clean_text(row.text).split()
        keys=[]
        if len(title.split())>=6:
            keys.append('title:'+title)
        if len(words)>=100:
            keys.append('prefix:'+' '.join(words[:80]))
        for key in keys:
            digest=hashlib.sha256(key.encode()).hexdigest()
            if digest in seen:
                parent[root(i)]=root(seen[digest])
            else:
                seen[digest]=i
    return [root(i) for i in range(len(frame))]

def bootstrap_f1(y,pred,repeats=1000,seed=42):
    y,pred=np.asarray(y),np.asarray(pred)
    rng=np.random.default_rng(seed)
    groups=[np.flatnonzero(y==label) for label in np.unique(y)]
    values=[]
    for _ in range(repeats):
        ids=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        values.append(f1_score(y[ids],pred[ids],average='macro',zero_division=0))
    low,high=np.quantile(values,[.025,.975])
    return dict(low=float(low),high=float(high),repeats=repeats,seed=seed,method='stratified article bootstrap; conditional on fitted model')
