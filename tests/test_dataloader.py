import unittest
import src.dataloader as dataloader
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import src.utils as utils

class TestDataloaderMethods(unittest.TestCase):
    def test_process_data_separates_data(self):
        X, Y = dataloader.process_data()

        path = os.path.join(os.path.dirname(__file__), "../data/adult.csv")
        df = pd.read_csv(path)
        updated_df = pd.get_dummies(df).drop_duplicates()
        dataset = updated_df.to_numpy()

        # checks the last column of X is the last column of the df (not including income)
        self.assertTrue((updated_df['native-country_Yugoslavia'].to_numpy() == X[:,-1]).all())

        # checks the shape of X is the same as the df without the income columns
        self.assertEqual(X.shape, (dataset.shape[0], dataset.shape[1] - 2))

    def test_normalize(self):
        train_X = np.array([[10, 5, 2], [6, 2, 9], [8, 10, 4]])
        train_y = np.array([1, 1, 0])

        test_X = np.array([[1, 90, 40], [2, 5, 5], [80, 17, 14]])
        test_y = np.array([1, 0, 0])

        (result_train_X_scaled, result_train_y), (result_test_X_scaled, result_test_y) = dataloader.normalize(
                (train_X, train_y),
                (test_X, test_y)
                )

        # write test for test_X is scaled according to train

        self.assertTrue((result_train_y == train_y).all())
        self.assertTrue((result_test_y == test_y).all())

    def test_loaders_returns_dataloaders(self):
        X, Y = dataloader.process_data()
        d1, d2 = utils.subsample(X, Y, 2500, 2)

        model_sets = dataloader.get_loaders([(d1, d2)], batch_size=8)

        d1_loader, d2_loader = model_sets[0]

        # returns a batch size of 8
        self.assertEqual(len(next(iter(d1_loader))[0]), 8)
        self.assertEqual(len(next(iter(d2_loader))[0]), 8)

if __name__ == '__main__':
    unittest.main()
