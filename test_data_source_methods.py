import unittest
from utils.data_source import DataSource

class TestDataSourceMethods(unittest.TestCase):
    def setUp(self):
        self.ds = DataSource(features_dir='features/features_20', features_length=1024, seq_length=20, model='densenet121')

    def test_unknown_model(self):
        with self.assertRaises(FileNotFoundError):
            DataSource(model='base_10')

    def test_default_args(self):
        self.assertEqual(self.ds.seq_length, 20)
        self.assertEqual(self.ds.features_dir, 'features/features_20')
        self.assertEqual(self.ds.features_length, 1024)
        self.assertEqual(self.ds.model, 'densenet121')

    def test_get_data_from_fold_method(self):
        filename, predictions = self.ds.get_data_from_fold('0', 'train') 
        self.assertEqual(len(filename), 149)
        self.assertEqual(len(predictions), 149)

        filename, predictions = self.ds.get_data_from_fold('0', 'test') 
        self.assertEqual(len(filename), 36)
        self.assertEqual(len(predictions), 36)

        filename, predictions = self.ds.get_data_from_fold('1', 'train') 
        self.assertEqual(len(filename), 149)
        self.assertEqual(len(predictions), 149)

        filename, predictions = self.ds.get_data_from_fold('1', 'test') 
        self.assertEqual(len(filename), 36)
        self.assertEqual(len(predictions), 36)

        filename, predictions = self.ds.get_data_from_fold('2', 'train') 
        self.assertEqual(len(filename), 148)
        self.assertEqual(len(predictions), 148)

        filename, predictions = self.ds.get_data_from_fold('2', 'test') 
        self.assertEqual(len(filename), 37)
        self.assertEqual(len(predictions), 37)

        filename, predictions = self.ds.get_data_from_fold('3', 'train') 
        self.assertEqual(len(filename), 148)
        self.assertEqual(len(predictions), 148)

        filename, predictions = self.ds.get_data_from_fold('3', 'test') 
        self.assertEqual(len(filename), 37)
        self.assertEqual(len(predictions), 37)

        filename, predictions = self.ds.get_data_from_fold('4', 'train') 
        self.assertEqual(len(filename), 146)
        self.assertEqual(len(predictions), 146)

        filename, predictions = self.ds.get_data_from_fold('4', 'test') 
        self.assertEqual(len(filename), 39)
        self.assertEqual(len(predictions), 39)

    def test_get_train_from_fold_method(self):
        x, y = self.ds.get_train_from_fold('0')
        self.assertEqual(x.shape, (149, 20, 1024))
        self.assertEqual(y.shape, (149, 3))

        x, y = self.ds.get_train_from_fold('1')
        self.assertEqual(x.shape, (149, 20, 1024))
        self.assertEqual(y.shape, (149, 3))

        x, y = self.ds.get_train_from_fold('2')
        self.assertEqual(x.shape, (148, 20, 1024))
        self.assertEqual(y.shape, (148, 3))

        x, y = self.ds.get_train_from_fold('3')
        self.assertEqual(x.shape, (148, 20, 1024))
        self.assertEqual(y.shape, (148, 3))

        x, y = self.ds.get_train_from_fold('4')
        self.assertEqual(x.shape, (146, 20, 1024))
        self.assertEqual(y.shape, (146, 3))

    def test_get_test_from_fold_method(self):
        x, y = self.ds.get_test_from_fold('0')
        self.assertEqual(x.shape, (36, 20, 1024))
        self.assertEqual(y.shape, (36, 3))

        x, y = self.ds.get_test_from_fold('1')
        self.assertEqual(x.shape, (36, 20, 1024))
        self.assertEqual(y.shape, (36, 3))

        x, y = self.ds.get_test_from_fold('2')
        self.assertEqual(x.shape, (37, 20, 1024))
        self.assertEqual(y.shape, (37, 3))

        x, y = self.ds.get_test_from_fold('3')
        self.assertEqual(x.shape, (37, 20, 1024))
        self.assertEqual(y.shape, (37, 3))

        x, y = self.ds.get_test_from_fold('4')
        self.assertEqual(x.shape, (39, 20, 1024))
        self.assertEqual(y.shape, (39, 3))

    def test_count_videos_method(self):
        result = self.ds.count_videos(fold_number_str='0', data_type='train')
        self.assertEqual(result.shape, (2,3))
        self.assertEqual(result[0][0], 'cov')
        self.assertEqual(result[0][1], 'pne')
        self.assertEqual(result[0][2], 'reg')
        self.assertEqual(result[1][0], '56')
        self.assertEqual(result[1][1], '40')
        self.assertEqual(result[1][2], '53')

if __name__ == '__main__':
    unittest.main()