import tempfile
import unittest
from pathlib import Path

import pandas as pd


class CoreTests(unittest.TestCase):
    def test_cleaning_removes_source_marker_and_html(self):
        from newscheck.data import clean_text
        self.assertEqual(clean_text('LONDON (Reuters) - <b>Hello</b> WORLD!'), 'hello world')

    def test_data_rejects_invalid_schema(self):
        from newscheck.data import prepare_data
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            pd.DataFrame({'wrong': ['x']}).to_csv(p/'Fake.csv', index=False)
            pd.DataFrame({'wrong': ['x']}).to_csv(p/'True.csv', index=False)
            with self.assertRaises(ValueError):
                prepare_data(p, p/'out', chunksize=2)

    def test_dedup_conflict_and_split(self):
        from newscheck.data import prepare_data
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            for label, name in [(0,'True.csv'),(1,'Fake.csv')]:
                rows = [{'title': f'Article {i}', 'text': f'Unique body class {label} number {i}', 'subject':'news', 'date':'January 1, 2017'} for i in range(50)]
                rows += [rows[0], {'title':'Conflict','text':'A conflicting label body','subject':'news','date':'bad'}, {'title':'Empty','text':'','subject':'news','date':''}]
                pd.DataFrame(rows).to_csv(p/name, index=False)
            df, meta = prepare_data(p, p/'out', chunksize=7)
            self.assertEqual(len(df), 100)
            self.assertEqual(meta['conflicting_groups'], 1)
            self.assertEqual(meta['empty_removed'], 2)
            self.assertEqual(df['text_hash'].nunique(), len(df))
            self.assertEqual(set(df['split']), {'train','validation','test'})
            df2, _ = prepare_data(p, p/'out2', chunksize=11)
            self.assertEqual(df['split'].tolist(), df2['split'].tolist())

    def test_models_do_not_learn_test_vocabulary(self):
        from newscheck.models import build_model
        x = ['verified official record agency statement', 'wild alien conspiracy secret rumor'] * 20
        y = [0, 1] * 20
        for name in ['baseline','bow','tfidf','lsa']:
            model = build_model(name)
            model.fit(x,y)
            self.assertEqual(len(model.predict(['unseenword report'])), 1)
            if name != 'baseline':
                self.assertNotIn('unseenword', model.named_steps['vectorizer'].vocabulary_)

    def test_sql_foreign_keys_and_queries(self):
        from newscheck.database import create_database, run_queries
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            df = pd.DataFrame([dict(article_id=1,title='A',text='news text',clean_text='news text',subject='news',date='2017-01-01',label=0,text_hash='abc',split='test',word_count=2)])
            con = create_database(p/'news.db', df)
            con.execute("INSERT INTO experiments VALUES ('tfidf','{}',0.8,0.9,1,1,100,1)")
            con.execute("INSERT INTO predictions VALUES ('tfidf',1,0,0.1)")
            con.commit()
            tables = run_queries(con,p/'sql')
            self.assertGreaterEqual(len(tables), 8)
            self.assertEqual(tables['01_class_balance']['n'].sum(),1)
            self.assertEqual(tables['06_model_errors']['errors'].iloc[0],0)
            import sqlite3
            with self.assertRaises(sqlite3.IntegrityError):
                con.execute("INSERT INTO predictions VALUES ('missing',1,0,0.1)")
            con.close()


if __name__ == '__main__':
    unittest.main()
