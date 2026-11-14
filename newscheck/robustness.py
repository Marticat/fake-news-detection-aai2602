"""Secondary checks: split sensitivity and a chronological holdout."""
import argparse
import time
from pathlib import Path
import pandas as pd
from .data import split_data,json_write
from .models import build_model
from .worker import evaluate

def temporal_partitions(frame,cutoff):
    dates=pd.to_datetime(frame.date,errors='coerce')
    latest=dates.groupby(frame.group_id).transform('max')
    earliest=dates.groupby(frame.group_id).transform('min')
    known=dates.notna().groupby(frame.group_id).transform('all')
    return frame[known & (latest<cutoff)],frame[known & (earliest>=cutoff)]

def run_checks(run_dir):
    run=Path(run_dir)
    frame=pd.read_csv(run/'articles.csv',keep_default_na=False)
    results=[]
    for seed in (7,21,84):
        split=split_data(frame,seed)
        train=split[split.split=='train'];test=split[split.split=='test']
        for name in ('baseline','bow','tfidf','lsa'):
            print(f'Robustness seed {seed}: {name}',flush=True)
            model=build_model(name);start=time.perf_counter()
            model.fit(train.clean_text.tolist(),train.label)
            p=model.predict_proba(test.clean_text.tolist())
            results.append(dict(check='grouped_seed',seed=seed,model=name,train_n=len(train),test_n=len(test),seconds=time.perf_counter()-start,**evaluate(test.label,model.classes_[p.argmax(1)],p[:,1])))
    dates=pd.to_datetime(frame.date,errors='coerce')
    cutoff=dates.dropna().quantile(.8).normalize()
    train,test=temporal_partitions(frame,cutoff)
    if train.label.nunique()==2 and test.label.nunique()==2:
        for name in ('baseline','bow','tfidf','lsa'):
            print(f'Temporal: {name}',flush=True)
            model=build_model(name);model.fit(train.clean_text.tolist(),train.label)
            p=model.predict_proba(test.clean_text.tolist())
            results.append(dict(check='temporal',cutoff=str(cutoff.date()),model=name,train_n=len(train),test_n=len(test),excluded_n=len(frame)-len(train)-len(test),**evaluate(test.label,model.classes_[p.argmax(1)],p[:,1])))
    json_write(run/'robustness.json',results)
    pd.DataFrame(results).to_csv(run/'robustness.csv',index=False)
    return results

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    run_checks(parser.parse_args().run)
