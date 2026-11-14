# Notes for the defence

## What changed from the original demo

The research now uses the downloaded ISOT corpus: 44,898 input articles, 38,313 after cleaning. The original 240 synthetic rows remain only a software-test fixture. Their perfect scores are not research findings.

## Explain the result carefully

The model predicts the label assigned by this historical dataset. It does not consult evidence about an individual claim. High held-out quality can reflect topic and publisher style. A deliberate subject-only diagnostic achieves 1.0 macro-F1, demonstrating complete category-label confounding in this test partition. Subject/date are not model features, but text can indirectly encode the same distinctions.

## Why the split is stronger

Exact normalized-body duplicates are removed before splitting. Identical long titles and identical long body prefixes define connected families. Entire families go to one partition, preventing detected versions of the same story from appearing in both train and test. This heuristic is conservative, not perfect paraphrase detection.

## Why validation and test differ

Train fits vocabulary and models. Validation selects the method. Test reports final quality. Selecting by the highest test score would bias the estimate. The main split is seed 42; seeds 7, 21 and 84 are additional sensitivity checks. The temporal test uses later articles from the same corpus, not independent external news.

## What E5 contributes

E5 is a frozen pretrained transformer. It converts the first 128 tokens into a 384-dimensional vector using masked mean pooling and L2 normalization. Logistic regression learns labels from those vectors. LSA is a separate linear representation learned from training TF-IDF; it is not a neural transformer. E5 weights are external to the saved classifier.

## Metrics in plain language

Accuracy: fraction correct. Precision(fake): among flagged articles, fraction labelled fake. Recall(fake): among fake-labelled articles, fraction detected. Macro-F1 gives equal weight to real and fake class F1. ROC-AUC measures score ranking. The bootstrap interval measures uncertainty from resampling test articles conditional on the trained model; it is not a guarantee for future publishers.

## SQL demonstration

Open main.py, choose SQL export, and inspect query 08 (subject errors). Explain JOIN from predictions through articles to subjects, GROUP BY and errors/n. Show query 05 ROW_NUMBER and query 09 SUM OVER. Explain foreign keys and why predictions have composite key (model, article_id).

## Honest remaining issues

The source is a public ISOT mirror, with exact file hashes, not a verified byte-identical download of the requested Kaggle archive. The selected corpus is under the recommended size: obtain the instructor's agreement rather than inflating files. External publisher validation is still absent. Grouping misses some paraphrases; E5 truncates long articles. CPU timing is descriptive and affected by thread settings/concurrent activity.

Each team member must record what they actually did. Proposed tasks are not evidence of completed individual contributions.
