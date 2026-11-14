"""Frozen E5 embeddings, bounded batches and token budget; no fine tuning."""
import os
from pathlib import Path
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class E5Embedding(BaseEstimator,TransformerMixin):
    def __init__(self, batch_size=16, max_length=128):
        self.batch_size=batch_size
        self.max_length=max_length

    def _load(self):
        if hasattr(self,'encoder_'):
            return
        import torch
        from transformers import AutoModel,AutoTokenizer
        revision='614241f622f53c4eeff9890bdc4f31cfecc418b3'
        local=Path.home()/'.cache/huggingface/hub/models--intfloat--multilingual-e5-small/snapshots'/revision
        source=getattr(self,'source_',None)
        if source is None or (source!='intfloat/multilingual-e5-small' and not Path(source).exists()):
            source=os.environ.get('NEWSCHECK_E5_PATH',str(local) if local.exists() else 'intfloat/multilingual-e5-small')
        if Path(source).exists():
            from .data import file_hash
            names=('model.safetensors','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json')
            fingerprint={name:file_hash(Path(source)/name) for name in names}
            if hasattr(self,'encoder_fingerprint_') and fingerprint!=self.encoder_fingerprint_:
                raise ValueError('E5 weights differ from the fitted encoder. Restore the original pinned model.')
            self.encoder_fingerprint_=fingerprint
        self.source_=source
        self.revision_=revision
        kwargs={} if Path(source).exists() else {'revision':revision}
        torch.set_num_threads(4)
        self.tokenizer_=AutoTokenizer.from_pretrained(source,**kwargs)
        self.encoder_=AutoModel.from_pretrained(source,**kwargs).eval()

    def fit(self,X,y=None):
        self._load()
        return self

    def transform(self,X):
        import torch
        self._load()
        texts=list(X)
        result=[]
        with torch.inference_mode():
            for start in range(0,len(texts),self.batch_size):
                batch=self.tokenizer_(['query: '+str(x) for x in texts[start:start+self.batch_size]],padding=True,truncation=True,max_length=self.max_length,return_tensors='pt')
                hidden=self.encoder_(**batch).last_hidden_state
                mask=batch['attention_mask'][...,None]
                pooled=(hidden*mask).sum(1)/mask.sum(1)
                result.append(torch.nn.functional.normalize(pooled,p=2,dim=1).cpu().numpy())
                if start%1024==0:
                    print(f'E5 encoded {start}/{len(texts)}',flush=True)
        return np.concatenate(result).astype(np.float32)

    def __getstate__(self):
        return {k:v for k,v in self.__dict__.items() if k not in ('encoder_','tokenizer_')}
