
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
