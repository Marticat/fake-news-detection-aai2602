# Measured research results

Run: 20261114T154147Z_kaggle_62b2ee (real ISOT public mirror; not synthetic).

44,898 input records, 38,313 cleaned articles. Primary test: 5,754 articles.

Selected by validation macro-F1: TF-IDF. Test macro-F1: 0.987139; accuracy: 0.987313; false positives: 20; false negatives: 53. E5 test macro-F1: 0.950056.

run.json records the completed experiment, provenance and limitations. comparison.csv and metrics_*.json contain measured quality and resources. test_predictions.csv includes all 28,770 primary test predictions without full article bodies. split_assignments.csv records article hashes, detected families, labels, dates and partitions. robustness.* covers three additional group splits and a later-period holdout; audit.json contains metadata/title/body diagnostics.

The directory name contains kaggle because the original orchestration used that corpus-family tag. Acquisition was from the explicitly identified public ISOT mirror; byte identity with the requested Kaggle archive was not established.

Local verification: 12 tests passed; primary prediction metrics were recomputed, detected families were checked for partition overlap, SQLite foreign keys and 28,770 prediction rows were verified. An independent code review's important findings were corrected. Intermediate GitHub upload commits briefly failed while dependent files were being added; subsequent complete-code CI runs passed.
