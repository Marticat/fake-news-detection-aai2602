# Requirements coverage

| Requirement | Implementation and evidence |
|---|---|
| 5+ functions | data, database, models, worker, validation, robustness, audit and workflow modules |
| Lists/dictionaries | ingestion records, partitions, metrics, query definitions |
| if/elif/else | CLI menu and model factory |
| Loops | chunk ingestion, model workers, menu, SQL exports |
| File persistence | CSV, JSON, SQLite, joblib, PDF/PPTX |
| Error handling | missing files/schema, insufficient groups, incomplete run, worker failures |
| Interface | interactive console menu and command-line subcommands |
| Modularity / standard library | pathlib, json, datetime, sqlite3, subprocess, hashlib |
| Baseline + two approaches | baseline + BoW + TF-IDF + LSA; frozen E5 added; quality/time/RSS measured |
| Database and SQL | 4 normalized tables; ER diagram; 10 queries with JOIN, HAVING, CTE and windows; Pandas exports |
| Volume / memory | actual 44,898 rows / 116.37 MB; guideline unmet and explicitly disclosed; chunking and sparse float32 |
| Robustness | three extra group seeds; temporal holdout; bootstrap intervals; metadata/title/body diagnostics |
| Deliverables | GitHub source, README and source links; measured report; editable PPTX and PDF |

Instructor approval of the selected corpus-size exception and verified individual contribution statements remain external requirements. They cannot be supplied by software.
