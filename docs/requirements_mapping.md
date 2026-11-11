# Соответствие требованиям

| № | Требование | Реализация / доказательство |
|---|---|---|
| 1 | >=5 функций кроме main | clean_text, prepare_data, split_data, create_database, run_queries, build_model, evaluate, run_project и др. |
| 2 | list/dict | records, conflicts, partitions, QUERIES, список metrics |
| 3 | if/elif/else | main.menu, models.build_model |
| 4 | for/while | меню while; чанки, строки, модели и SQL в for |
| 5 | чтение/запись файлов | CSV, JSON, SQLite, joblib, опционально Parquet |
| 6 | try/except | CLI/menu, отсутствующие зависимости, неудачный run |
| 7 | интерфейс | консольное меню из 5 операций + выход |
| 8 | несколько .py и stdlib | 8 модулей; pathlib, datetime, json, sqlite3, subprocess |
| 9 | >=2 подхода и baseline | baseline + BoW + TF-IDF + LSA; дополнительный MiniLM; качество/время/RSS |
| 10 | SQL и ER | subjects/articles/experiments/predictions; 10 запросов, JOIN, HAVING, CTE, оконные функции; pd.read_sql_query |
| 11 | объём и память | manifest проверяет >=500k строк или >=200 MB; chunksize, dtype, sparse, optional Parquet; исключение при необходимости |

Обязательные материалы: код + README в проекте; источник данных в README; отчёт и PDF-презентация создаются автоматически. Удалённый GitHub и окончательные результаты на Kaggle требуют завершения загрузки данных/публикации.
