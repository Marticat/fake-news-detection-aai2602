"""Normalized SQLite storage with enforced foreign keys and analytical SQL."""
import sqlite3
import json
from pathlib import Path
import pandas as pd

SCHEMA = '''
PRAGMA foreign_keys=ON;
CREATE TABLE subjects(subject_id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
CREATE TABLE articles(
 article_id INTEGER PRIMARY KEY, subject_id INTEGER NOT NULL REFERENCES subjects(subject_id),
 title TEXT NOT NULL, text TEXT NOT NULL, clean_text TEXT NOT NULL, date TEXT,
 label INTEGER NOT NULL CHECK(label IN(0,1)), text_hash TEXT NOT NULL UNIQUE,
 split TEXT NOT NULL CHECK(split IN('train','validation','test')), word_count INTEGER NOT NULL);
CREATE TABLE experiments(
 model TEXT PRIMARY KEY, config_json TEXT NOT NULL, validation_f1 REAL, test_f1 REAL,
 fit_seconds REAL, predict_seconds REAL, peak_rss_mib REAL, model_mib REAL);
CREATE TABLE predictions(
 model TEXT NOT NULL REFERENCES experiments(model), article_id INTEGER NOT NULL REFERENCES articles(article_id),
 predicted INTEGER NOT NULL CHECK(predicted IN(0,1)), probability_fake REAL CHECK(probability_fake BETWEEN 0 AND 1),
 PRIMARY KEY(model,article_id));
CREATE INDEX idx_articles_split ON articles(split);
CREATE INDEX idx_articles_subject ON articles(subject_id);
'''

QUERIES = {
'01_class_balance': '''SELECT label, COUNT(*) AS n, ROUND(AVG(word_count),1) AS mean_words FROM articles GROUP BY label''',
'02_subjects': '''SELECT s.name, a.label, COUNT(*) AS n FROM articles a JOIN subjects s USING(subject_id) GROUP BY s.name,a.label HAVING COUNT(*)>=5 ORDER BY n DESC''',
'03_monthly': '''SELECT substr(date,1,7) AS month,label,COUNT(*) AS n FROM articles WHERE date IS NOT NULL GROUP BY month,label ORDER BY month,label''',
'04_long_articles': '''SELECT article_id,title,word_count FROM articles WHERE word_count>(SELECT AVG(word_count) FROM articles) ORDER BY word_count DESC LIMIT 20''',
'05_rank_by_subject': '''WITH ranked AS (SELECT s.name,a.title,a.word_count,ROW_NUMBER() OVER(PARTITION BY a.subject_id ORDER BY a.word_count DESC,a.article_id) AS rank FROM articles a JOIN subjects s USING(subject_id)) SELECT * FROM ranked WHERE rank<=3''',
'06_model_errors': '''SELECT p.model,COUNT(*) AS evaluated,SUM(p.predicted<>a.label) AS errors,ROUND(AVG(p.predicted=a.label),4) AS accuracy FROM predictions p JOIN articles a USING(article_id) GROUP BY p.model''',
'07_confusion': '''SELECT p.model,a.label AS actual,p.predicted,COUNT(*) AS n FROM predictions p JOIN articles a USING(article_id) GROUP BY p.model,a.label,p.predicted ORDER BY p.model,a.label,p.predicted''',
'08_error_by_subject': '''WITH totals AS (SELECT p.model,s.name,COUNT(*) AS n,SUM(p.predicted<>a.label) AS errors FROM predictions p JOIN articles a USING(article_id) JOIN subjects s USING(subject_id) GROUP BY p.model,s.name) SELECT *,ROUND(1.0*errors/n,4) AS error_rate FROM totals ORDER BY error_rate DESC''',
'09_cumulative_volume': '''WITH monthly AS (SELECT substr(date,1,7) AS month,COUNT(*) AS n FROM articles WHERE date IS NOT NULL GROUP BY month) SELECT month,n,SUM(n) OVER(ORDER BY month ROWS UNBOUNDED PRECEDING) AS cumulative_n FROM monthly ORDER BY month''',
'10_model_ranking': '''SELECT model,validation_f1,test_f1,fit_seconds,peak_rss_mib,DENSE_RANK() OVER(ORDER BY validation_f1 DESC) AS validation_rank FROM experiments ORDER BY validation_rank,model'''
}


def create_database(path, df):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to replace existing database: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    mapping = {name:i+1 for i,name in enumerate(sorted(df.subject.unique()))}
    con.executemany('INSERT INTO subjects VALUES (?,?)', [(v,k) for k,v in mapping.items()])
    rows = [(int(r.article_id),mapping[r.subject],r.title,r.text,r.clean_text,r.date,int(r.label),r.text_hash,r.split,int(r.word_count)) for r in df.itertuples()]
    con.executemany('INSERT INTO articles VALUES (?,?,?,?,?,?,?,?,?,?)',rows)
    con.commit()
    return con


def run_queries(con, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    results = {}
    manifest=out_dir.parent/'run.json'
    provenance=json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else None
    for name,sql in QUERIES.items():
        frame = pd.read_sql_query(sql,con)
        if provenance:
            frame['dataset_kind']=provenance['dataset_kind']
            frame['run_id']=out_dir.parent.name
            frame['artifact_status']='sql_exported'
        frame.to_csv(out_dir/f'{name}.csv',index=False)
        results[name] = frame
    (out_dir/'queries.sql').write_text('\n\n'.join(f'-- {n}\n{s};' for n,s in QUERIES.items()),encoding='utf-8')
    (out_dir/'schema.sql').write_text(SCHEMA,encoding='utf-8')
    return results
