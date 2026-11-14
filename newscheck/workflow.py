"""Run orchestration; incomplete runs never masquerade as completed experiments."""
import json
import os
import platform
import sqlite3
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd
import sklearn
import numpy as np
from .data import prepare_data,create_demo,clean_text,json_write
from .database import create_database,run_queries


def run_project(raw_dir='data/raw',runs_dir='runs',demo=False,include_minilm=False,isolated=True,documents=True,include_e5=False):
    kind='synthetic_demo' if demo else 'kaggle'
    if not demo and (Path(raw_dir)/'SYNTHETIC_ONLY.txt').exists():
        raise ValueError('Synthetic fixture cannot be used as a real-data run. Use --demo.')
    run=Path(runs_dir).resolve()/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+kind+'_'+uuid4().hex[:6])
    run.mkdir(parents=True)
    if demo:
        raw_dir=run/'synthetic_input'
        create_demo(raw_dir)
    meta=dict(status='running',dataset_kind=kind,run_id=run.name,created_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version,platform=platform.platform(),sklearn=sklearn.__version__,pandas=pd.__version__,numpy=np.__version__,
        isolated_workers=isolated,thread_limit=1 if isolated else None,metrics=[],selected_model=None,
        model_thread_limits={'core':1 if isolated else None,'e5':4},
        limitations=['Text classification, not fact verification.','English only; historical source and topic bias.',
        'Exact body deduplication and title/prefix family grouping; semantic near-duplicates may remain.',
        'Main split seed 42; article bootstrap intervals are conditional on fitted models. Separate robustness checks are optional.',
        'Peak RSS sampled every 20 ms; includes data and interpreter. Short peaks can be missed.',
        'All cleaned texts are retained in RAM for batch model training. Ingestion is chunked, training is not out-of-core.'])
    json_write(run/'run.json',meta)
    con=None
    try:
        df,data=prepare_data(raw_dir,run,dataset_kind=kind)
        meta['data']=data
        con=create_database(run/'news.db',df)
        models=['baseline','bow','tfidf','lsa']+(['minilm'] if include_minilm else [])+(['e5'] if include_e5 else [])
        for name in models:
            print(f'Training {name}...',flush=True)
            if isolated:
                env=os.environ.copy()
                for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
                    env[key]='1'
                log=run/f'worker_{name}.log'
                try:
                    with log.open('w',encoding='utf-8') as output:
                        subprocess.run([sys.executable,'-m','newscheck.worker','--run',str(run),'--model',name],
                            cwd=Path(__file__).resolve().parents[1],env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(f'{name} training failed. Details: {log}') from exc
                row=json.loads((run/f'metrics_{name}.json').read_text(encoding='utf-8'))
            else:
                from .worker import train_worker
                row=train_worker(run,name)
            meta['metrics'].append(row)
            con.execute('INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?)',(name,json.dumps(row['config']),row['validation']['macro_f1'],row['test']['macro_f1'],row['fit_seconds'],row['predict_seconds'],row['peak_rss_mib'],row['model_mib']))
            pred=pd.read_csv(run/f'predictions_{name}.csv')
            con.executemany('INSERT INTO predictions VALUES (?,?,?,?)',[(name,int(r.article_id),int(r.predicted),float(r.probability_fake)) for r in pred.itertuples()])
            con.commit()
            json_write(run/'run.json',meta)
        # The test scores do not participate in model selection.
        meta['selected_model']=max(meta['metrics'],key=lambda r:r['validation']['macro_f1'])['model']
        run_queries(con,run/'sql')
        pd.DataFrame([dict(dataset_kind=kind,run_id=run.name,artifact_status='analysis_complete',model=r['model'],validation_macro_f1=r['validation']['macro_f1'],**{k:v for k,v in r['test'].items() if k!='confusion_matrix'},fit_seconds=r['fit_seconds'],predict_seconds=r['predict_seconds'],peak_rss_mib=r['peak_rss_mib'],model_mib=r['model_mib']) for r in meta['metrics']]).to_csv(run/'comparison.csv',index=False)
        from .charts import make_charts
        make_charts(run,df,meta)
        meta['status']='complete'
        json_write(run/'run.json',meta)
        if documents:
            from .reporting import build_documents
            build_documents(run)
        print(f'Completed: {run}',flush=True)
        return run
    except (Exception,KeyboardInterrupt) as exc:
        meta['status']='failed'
        meta['error']=f'{type(exc).__name__}: {exc}'
        json_write(run/'run.json',meta)
        raise
    finally:
        if con is not None:
            con.close()


def predict_text(run_dir,text):
    run_dir=Path(run_dir)
    text=clean_text(text)
    if len(text.split())<3:
        raise ValueError('Enter at least three words of English article text.')
    meta=json.loads((run_dir/'run.json').read_text(encoding='utf-8'))
    if meta['status']!='complete':
        raise ValueError('This run is incomplete; choose a completed run.')
    model=joblib.load(run_dir/'models'/f"{meta['selected_model']}.joblib")
    probability=float(model.predict_proba([text])[0,1])
    label=int(model.predict([text])[0])
    return dict(label='fake' if label else 'real',probability_fake=probability,
        model=meta['selected_model'],dataset_kind=meta['dataset_kind'],
        note='Model score is not a verified probability that the claim is false.')


def export_sql(run_dir):
    run_dir=Path(run_dir)
    db=run_dir/'news.db'
    if not db.exists():
        raise FileNotFoundError(db)
    with sqlite3.connect(db) as con:
        return run_queries(con,run_dir/'sql')
