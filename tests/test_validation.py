import unittest
import pandas as pd
from newscheck.data import split_data

class ValidationTests(unittest.TestCase):
    def test_temporal_excludes_families_with_unknown_dates(self):
        from newscheck.robustness import temporal_partitions
        frame=pd.DataFrame({'date':['2017-01-01','','2018-01-01','2016-01-01'], 'group_id':[1,1,2,3]})
        train,test=temporal_partitions(frame,pd.Timestamp('2017-06-01'))
        self.assertEqual(train.index.tolist(),[3])
        self.assertEqual(test.index.tolist(),[2])

    def test_neural_serialization_keeps_fitted_identity(self):
        from newscheck.neural import E5Embedding
        model=E5Embedding()
        model.source_='fixed-source'
        model.encoder_='not serialized'
        state=model.__getstate__()
        self.assertEqual(state['source_'],'fixed-source')
        self.assertNotIn('encoder_',state)

    def test_fingerprinted_encoder_cannot_fall_back_unverified(self):
        from newscheck.neural import E5Embedding
        model=E5Embedding()
        model.source_='intfloat/multilingual-e5-small'
        model.encoder_fingerprint_={'model.safetensors':'known-hash'}
        with self.assertRaisesRegex(ValueError,'identity can be verified'):
            model._load()

    def test_groups_never_cross_partitions(self):
        df = pd.DataFrame({'label':[i%2 for i in range(200)], 'group_id':[i//2 for i in range(200)]})
        result = split_data(df)
        self.assertEqual(result.groupby('group_id')['split'].nunique().max(),1)

    def test_bootstrap_bounds_and_reproducibility(self):
        from newscheck.validation import bootstrap_f1
        result = bootstrap_f1([0,0,1,1],[0,1,1,1], repeats=100)
        self.assertTrue(0 <= result['low'] <= result['high'] <= 1)
        self.assertEqual(result,bootstrap_f1([0,0,1,1],[0,1,1,1],repeats=100))
