"""Comparable representations, all fitted only on the training partition."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer


class SafeSVD(BaseEstimator,TransformerMixin):
    def __init__(self,n_components=128,random_state=42):
        self.n_components=n_components
        self.random_state=random_state

    def fit(self,X,y=None):
        n = min(self.n_components, X.shape[0]-1, X.shape[1]-1)
        if n < 1:
            raise ValueError('Not enough features for LSA embeddings.')
        self.svd_ = TruncatedSVD(n_components=n,random_state=self.random_state)
        self.svd_.fit(X)
        return self

    def transform(self,X):
        return self.svd_.transform(X).astype(np.float32)


class SentenceEmbedding(BaseEstimator,TransformerMixin):
    """Frozen MiniLM encoder. Long documents are split into token chunks and averaged."""
    def __init__(self,model_name='sentence-transformers/all-MiniLM-L6-v2',batch_size=32):
        self.model_name=model_name
        self.batch_size=batch_size

    def fit(self,X,y=None):
        self._load()
        return self

    def _load(self):
        if not hasattr(self,'encoder_'):
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError('Install requirements-embeddings.txt to run MiniLM.') from exc
            self.encoder_=SentenceTransformer(self.model_name,device='cpu')

    def transform(self,X):
        self._load()
        vectors=[]
        tokenizer=self.encoder_.tokenizer
        width=max(8,self.encoder_.max_seq_length-2)
        for text in X:
            ids=tokenizer.encode(str(text),add_special_tokens=False)
            chunks=[tokenizer.decode(ids[i:i+width],skip_special_tokens=True) for i in range(0,len(ids),width)] or ['']
            embeddings=self.encoder_.encode(chunks,batch_size=self.batch_size,convert_to_numpy=True,show_progress_bar=False)
            vector=embeddings.mean(axis=0)
            vectors.append(vector/max(np.linalg.norm(vector),1e-12))
        return np.asarray(vectors,dtype=np.float32)

    def __getstate__(self):
        state=self.__dict__.copy()
        state.pop('encoder_',None)
        return state


def build_model(name):
    lr=LogisticRegression(C=1.0,max_iter=2000,solver='liblinear',random_state=42)
    common=dict(max_features=30000,ngram_range=(1,2),min_df=2,dtype=np.float32)
    if name=='baseline':
        return DummyClassifier(strategy='most_frequent')
    elif name=='bow':
        return Pipeline([('vectorizer',CountVectorizer(**common)),('normalize',Normalizer()),('classifier',lr)])
    elif name=='tfidf':
        return Pipeline([('vectorizer',TfidfVectorizer(sublinear_tf=True,**common)),('classifier',lr)])
    elif name=='lsa':
        return Pipeline([('vectorizer',TfidfVectorizer(sublinear_tf=True,**common)),('svd',SafeSVD()),('normalize',Normalizer()),('classifier',lr)])
    elif name=='minilm':
        return Pipeline([('embedding',SentenceEmbedding()),('classifier',lr)])
    elif name=='e5':
        from .neural import E5Embedding
        return Pipeline([('embedding',E5Embedding()),('classifier',lr)])
    raise ValueError(f'Unknown model: {name}')
