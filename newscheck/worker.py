"""Isolated model benchmark: full-process RSS sampled every 20 ms."""
import argparse
import json
import platform
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
from .data import json_write
from .models import build_model


class MemoryMonitor:
    def __enter__(self):
        self.process=psutil.Process()
        self.start=self.peak=self.process.memory_info().rss
        self.stop=threading.Event()
        self.thread=threading.Thread(target=self._sample,daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.stop.wait(0.02):
            self.peak=max(self.peak,self.process.memory_info().rss)

    def __exit__(self,*args):
        self.peak=max(self.peak,self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()


def evaluate(y,pred,proba):
    return dict(accuracy=float(accuracy_score(y,pred)),precision_fake=float(precision_score(y,pred,zero_division=0)),
        recall_fake=float(recall_score(y,pred,zero_division=0)),macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),
        roc_auc=float(roc_auc_score(y,proba)),confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())


def train_worker(run_dir,model_name):
    run_dir=Path(run_dir)
    provenance=json.loads((run_dir/'run.json').read_text(encoding='utf-8'))
    tags=dict(dataset_kind=provenance['dataset_kind'],run_id=run_dir.name,artifact_status='model_complete')
    with MemoryMonitor() as memory:
        frame=pd.read_csv(run_dir/'articles.csv',keep_default_na=False)
        partitions={name:frame.loc[frame.split==name] for name in ['train','validation','test']}
        train,val,test=(partitions[name] for name in ['train','validation','test'])
        model=build_model(model_name)
        start=time.perf_counter()
        model.fit(train.clean_text.tolist(),train.label.to_numpy())
        fit_seconds=time.perf_counter()-start
        val_proba=model.predict_proba(val.clean_text.tolist())[:,1]
        val_pred=model.classes_[np.argmax(model.predict_proba(val.clean_text.tolist()),axis=1)]
        validation=evaluate(val.label,val_pred,val_proba)
        start=time.perf_counter()
        probabilities=model.predict_proba(test.clean_text.tolist())
        predict_seconds=time.perf_counter()-start
        predicted=model.classes_[np.argmax(probabilities,axis=1)]
        metrics=evaluate(test.label,predicted,probabilities[:,1])
        folder=run_dir/'models'
        folder.mkdir(exist_ok=True)
        model_path=folder/f'{model_name}.joblib'
        joblib.dump(model,model_path,compress=3)
        pred=test[['article_id','title','text','subject','label','word_count']].copy()
        pred['predicted']=predicted
        pred['probability_fake']=probabilities[:,1]
        pred['error_type']=np.where(pred.label==pred.predicted,'correct',np.where(pred.predicted==1,'false_positive','false_negative'))
        for key,value in tags.items():
            pred[key]=value
        pred.to_csv(run_dir/f'predictions_{model_name}.csv',index=False)
        errors=pred.loc[pred.label!=pred.predicted].copy()
        errors['wrong_confidence']=np.where(errors.predicted==1,errors.probability_fake,1-errors.probability_fake)
        errors.sort_values('wrong_confidence',ascending=False).to_csv(run_dir/f'errors_{model_name}.csv',index=False)
        pred['length_bucket']=pd.cut(pred.word_count,[0,100,300,1000,float('inf')],labels=['1-100','101-300','301-1000','1001+'])
        pred['error']=(pred.label!=pred.predicted).astype(int)
        lengths=pred.groupby('length_bucket',observed=True).agg(n=('article_id','size'),error_rate=('error','mean'))
        for key,value in tags.items():
            lengths[key]=value
        lengths.to_csv(run_dir/f'length_errors_{model_name}.csv')
        if model_name in ['bow','tfidf']:
            terms=model.named_steps['vectorizer'].get_feature_names_out()
            weights=model.named_steps['classifier'].coef_[0]
            order=np.r_[np.argsort(weights)[:20],np.argsort(weights)[-20:]]
            pd.DataFrame({'term':terms[order],'coefficient_toward_fake':weights[order],**tags}).to_csv(run_dir/f'terms_{model_name}.csv',index=False)
    result=dict(**tags,model=model_name,validation=validation,test=metrics,fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,predict_ms_per_article=1000*predict_seconds/len(test),
        peak_rss_mib=memory.peak/(1024**2),rss_increase_mib=(memory.peak-memory.start)/(1024**2),
        model_mib=model_path.stat().st_size/(1024**2),train_n=len(train),validation_n=len(val),test_n=len(test),
        config={k:str(v) for k,v in model.get_params(deep=False).items()})
    json_write(run_dir/f'metrics_{model_name}.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--model',required=True)
    args=parser.parse_args()
    train_worker(args.run,args.model)
