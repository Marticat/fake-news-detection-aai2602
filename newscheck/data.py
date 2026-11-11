"""Chunked ingestion, conservative normalization and reproducible splitting."""
import hashlib
import html
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42


def clean_text(text):
    text = html.unescape(str(text))
    text = re.sub(r'^.{0,100}?\(Reuters\)\s*[-–—:]?\s*', '', text, flags=re.I)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\breuters\b', ' ', text, flags=re.I)
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', ' ', text.lower())).strip()


def json_write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def split_data(df, seed=SEED):
    counts = df.label.value_counts()
    if len(counts) != 2 or counts.min() < 10:
        raise ValueError('Need at least 10 unique, non-conflicting articles in each class.')
    train, rest = train_test_split(df.index, test_size=0.30, stratify=df.label, random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=df.loc[rest,'label'], random_state=seed)
    df = df.copy()
    df['split'] = 'train'
    df.loc[val,'split'] = 'validation'
    df.loc[test,'split'] = 'test'
    return df


def prepare_data(raw_dir, out_dir, chunksize=5000, seed=SEED, dataset_kind='kaggle'):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    if chunksize < 1:
        raise ValueError('chunksize must be positive')
    files = [(raw_dir/'True.csv',0),(raw_dir/'Fake.csv',1)]
    for path, _ in files:
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}. See README for dataset download.')
    records, conflicts = {}, set()
    total = empty = repeats = 0
    for path, label in files:
        required = {'title','text','subject','date'}
        if not required.issubset(pd.read_csv(path,nrows=0).columns):
            raise ValueError(f'{path.name}: required columns are {sorted(required)}')
        for chunk in pd.read_csv(path, chunksize=chunksize, dtype=str, keep_default_na=False):
            for row in chunk.to_dict('records'):
                total += 1
                body = clean_text(row['text'])
                if not body:
                    empty += 1
                    continue
                key = hashlib.sha256(body.encode('utf-8')).hexdigest()
                if key in records:
                    repeats += 1
                    if records[key]['label'] != label:
                        conflicts.add(key)
                    continue
                records[key] = dict(title=row['title'], text=row['text'],
                    clean_text=(clean_text(row['title'])+' '+body).strip(),
                    subject=row['subject'] or 'unknown', date=row['date'], label=label,
                    text_hash=key, word_count=len(body.split()))
    for key in conflicts:
        records.pop(key, None)
    if not records:
        raise ValueError('No usable articles remain after cleaning.')
    df = pd.DataFrame(records.values()).reset_index(drop=True)
    df['article_id'] = df.index + 1
    parsed = pd.to_datetime(df.date, errors='coerce', format='mixed')
    invalid_dates = int(parsed.isna().sum())
    df['date'] = parsed.dt.strftime('%Y-%m-%d').where(parsed.notna(), None)
    df['label'] = df.label.astype('int8')
    df['word_count'] = df.word_count.astype('int32')
    df = split_data(df, seed)
    df['dataset_kind']=dataset_kind
    df['run_id']=out_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir/'articles.csv',index=False)
    parquet = False
    try:
        df.to_parquet(out_dir/'articles.parquet',index=False)
        parquet = True
    except ImportError:
        pass
    size = sum(path.stat().st_size for path,_ in files)
    meta = dict(dataset_kind=dataset_kind, run_id=out_dir.name, artifact_status='data_prepared', raw_rows=total, clean_rows=len(df),
        empty_removed=empty, duplicate_rows=repeats, conflicting_groups=len(conflicts),
        invalid_dates=invalid_dates, raw_bytes=size, volume_threshold_met=(total>=500000 or size>=200_000_000),
        seed=seed, chunksize=chunksize, parquet_written=parquet,
        class_counts={str(k):int(v) for k,v in df.label.value_counts().items()},
        split_counts={str(k):int(v) for k,v in df.split.value_counts().items()},
        source_files={path.name:dict(sha256=file_hash(path),bytes=path.stat().st_size) for path,_ in files})
    json_write(out_dir/'data_manifest.json',meta)
    return df, meta


def create_demo(raw_dir):
    """Synthetic engineering fixture. Not evidence about news detection quality."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    subjects = ['economy','science','politics','health']
    for label,name in [(0,'True.csv'),(1,'Fake.csv')]:
        rows = []
        for i in range(120):
            subject = subjects[i%4]
            if label == 0:
                body = f'Official records describe a {subject} meeting. The agency published a report with figures and evidence. Investigators reviewed case {i} and scheduled a public hearing.'
            else:
                body = f'A viral rumor claims a secret {subject} conspiracy. Anonymous sources say aliens control everything. No evidence supports sensational claim number {i} shared online.'
            rows.append(dict(title=f'{subject.title()} update {i}',text=body,subject=subject,date=f'2017-01-{i%28+1:02d}'))
        pd.DataFrame(rows).to_csv(raw_dir/name,index=False)
    (raw_dir/'SYNTHETIC_ONLY.txt').write_text('Synthetic software test fixture. Not Kaggle data. Do not use metrics as research findings.',encoding='utf-8')
