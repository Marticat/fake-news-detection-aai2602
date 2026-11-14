"""Diagnostic ablations on the fixed primary split (not model selection)."""
import argparse
from pathlib import Path
import pandas as pd
from .data import clean_text,json_write
from .models import build_model
from .worker import evaluate

def run_audit(run_dir):
    run=Path(run_dir);df=pd.read_csv(run/'articles.csv',keep_default_na=False)
    train=df[df.split=='train'];test=df[df.split=='test'];results=[]
    # Deliberately metadata-only: exposes confounding; never a deployed model.
    lookup=train.groupby('subject').label.mean()
    probabilities=test.subject.map(lookup).fillna(float(train.label.mean())).to_numpy()
    results.append(dict(check='subject_only_diagnostic',**evaluate(test.label,(probabilities>=.5).astype(int),probabilities)))
    for column in ('title','text'):
        print('Ablation '+column,flush=True)
        model=build_model('tfidf')
        model.fit(train[column].map(clean_text).tolist(),train.label)
        p=model.predict_proba(test[column].map(clean_text).tolist())
        results.append(dict(check='tfidf_'+column+'_only',**evaluate(test.label,model.classes_[p.argmax(1)],p[:,1])))
    json_write(run/'audit.json',results)
    return results

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    run_audit(parser.parse_args().run)
