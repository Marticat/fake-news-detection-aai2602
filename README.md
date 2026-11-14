# Fake News Detection — AAI-2602

Kurmet Adilzhan · Tolegenov Nursultan · Issakulov Zeiin
Programming for Data Analysis and Database | Applied Artificial Intelligence

A modular Python project combining news-text classification, SQLite analytics and reproducible evaluation. The current research execution uses real ISOT CSV files, not the earlier synthetic software fixture.

## Run

Python 3.10+:

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

The interactive menu supports training, classification, SQL export and document generation. For a quick software-only test: `python main.py run --demo`.

## Data

Requested source: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Executed public ISOT mirror: https://github.com/emilurosev/isot-fake-news-dataset/tree/aa8ca45d9da5a308479c52d121c42f36a1f97e69

Download `True.csv` and `Fake.csv` with the GitHub **Download raw file** button; put them in `data/raw/`. See [docs/SOURCE.md](docs/SOURCE.md) for provenance. The executed files contain 44,898 rows and 116,372,816 bytes; after 715 empty bodies and 5,870 repeated bodies are removed, 38,313 articles remain. Hashes are stored in every run's `data_manifest.json`. Byte identity against the requested Kaggle archive was not verified.

This selected corpus is below the recommended 500,000 rows or 200 MB. An instructor-approved exception remains necessary if that recommendation is enforced. No artificial duplication is used to inflate volume. Raw data and model weights are not committed; use the source links and applicable source licenses.

## Research execution

```bash
python main.py run --raw data/raw
# Frozen neural representation (additional dependencies and pretrained weights):
python -m pip install -r requirements-e5.txt
python main.py run --raw data/raw --include-e5
python -m newscheck.robustness --run runs/RUN_ID
python -m newscheck.audit --run runs/RUN_ID
python main.py predict --run runs/RUN_ID --text "An English news article to classify"
```

E5 uses `intfloat/multilingual-e5-small`, pinned revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`. A fresh machine downloads the encoder from Hugging Face. Set `NEWSCHECK_E5_PATH` to a local pretrained model directory for offline execution. The joblib classifier does **not** bundle neural weights. Optional MiniLM remains available with `--include-minilm` and `requirements-embeddings.txt`.

| Method | Representation | Classifier |
|---|---|---|
| Baseline | No text features | Most frequent training class |
| BoW | Word/bigram counts, L2 | Logistic Regression |
| TF-IDF | Sublinear TF-IDF word/bigrams | Logistic Regression |
| LSA | TF-IDF, up to 128 SVD dimensions, L2 | Logistic Regression |
| E5 | Frozen 384-dimensional neural vectors | Logistic Regression |

E5 uses a query prefix, masked mean pooling, L2 normalization, batches of 16 and the first 128 tokenizer tokens. Lexical methods use full cleaned text. This is a comparison of practical pipelines with different input budgets. The neural encoder is not fine-tuned on news labels.

## Evaluation safeguards

- Exact normalized bodies are deduplicated before splitting; contradictory labels are excluded.
- Identical normalized titles of at least six words, or identical 80-word prefixes in bodies of at least 100 words, define connected duplicate families. Detected families never cross partitions. This is not exhaustive semantic deduplication.
- GroupShuffleSplit approximates 70/15/15, seed 42. Counts and class composition are observed, not assumed. Synthetic smoke tests use a stratified split.
- Vocabulary, IDF, SVD and classifier fit on training only. Validation macro-F1 selects the model; test data does not select it.
- Accuracy, fake precision/recall, macro-F1, ROC-AUC and confusion matrices are exported. Conditional 95% F1 intervals use 1,000 stratified article bootstrap draws.
- Additional seeds 7/21/84 and an 80th-percentile chronological holdout measure lexical-model sensitivity. Date-spanning families are excluded from the temporal comparison.
- `audit.json` includes subject-only confounding, title-only and body-only diagnostics. Metadata is never fed to deployed text models.

Remaining scientific limits: historical English corpus, source/topic shortcuts, no independent external publisher test, heuristic families, E5 truncation and conditional rather than full training-uncertainty intervals. Model scores are not calibrated factual-truth probabilities.

## SQL and software structure

`newscheck/data.py` handles chunked ingestion and cleaning; `models.py` and `neural.py` define representations; `worker.py` measures quality/resources; `workflow.py` orchestrates runs; `database.py` implements SQLite; `validation.py`, `robustness.py`, and `audit.py` evaluate uncertainty and sensitivity; `reporting.py` generates PDFs.

Four tables: `subjects`, `articles`, `experiments`, `predictions`. Ten SQL queries include JOIN, GROUP BY/HAVING, subquery, CTE, ROW_NUMBER, SUM OVER and DENSE_RANK. Results load into Pandas and export to CSV. ER diagrams and schema/query sources are included.

CSV reads use 5,000-row chunks, compact label/length types and sparse float32 features. Cleaned text remains in RAM, so training is not out-of-core. Optional Parquet output requires pyarrow. Process RSS is sampled every 20 ms. Core workers use one numerical thread and E5 four; concurrent machine activity affects timings. Saved E5 joblib size excludes its 470.64 MB encoder.

## Verification and outputs

```bash
python -m unittest discover -s tests -v
python main.py sql --run runs/RUN_ID
python main.py report --run runs/RUN_ID
```

Each run stores manifests, metrics, model files, predictions, real errors, feature coefficients, length diagnostics, SQL CSVs, a SQLite database and charts. Failed runs are marked and rejected for inference. Load only trusted joblib files.

The research report and editable presentation are in `docs/`. Measured tables and provenance are in `results/`; large raw files and trained models are reproduced locally. Earlier artifacts with DEMO in their filenames are software-validation records only; use the new report and presentation without DEMO for the real experiment.

## Team contribution record

The three students are project authors. Actual individual contributions must be supplied by the team before submission; no guessed allocation is represented as completed work.

## Rebuild designed PDF / PPTX deliverables

`python tools/build_deliverables.py runs/RUN_ID` after the main run, robustness and audit are complete. Outputs are in `outputs/Desktop_Delivery/`. The editable PPTX writer is included and needs no PowerPoint installation. Newspaper photograph: Mr Cup / Fabien Barral, https://unsplash.com/photos/Mwuod2cm8g4, Unsplash License.
