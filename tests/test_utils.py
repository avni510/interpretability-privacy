from src import utils as utils
import unittest
import pandas as pd
import os
import numpy as np

path = os.path.join(os.path.dirname(__file__), "../data/adult.csv")
df = pd.read_csv(path)

updated_df = pd.get_dummies(df)
updated_df = updated_df.drop_duplicates()
dataset = updated_df.to_numpy()

Y = dataset[:, -1]
X = np.delete(dataset, -1, axis = 1)
X = np.delete(X, -1, axis = 1)

class TestUtilsMethods(unittest.TestCase):
    def test_subsample_returns_subset_length(self):
        subsamples = utils.subsample(X, Y, 2500, 2, is_disjoint=True)
        self.assertEqual(len(subsamples), 2)

        (d1_X, d1_y), (d2_X, d2_y) = subsamples[0], subsamples[1]

        self.assertEqual(d1_X.shape, (2500, X.shape[1]))
        self.assertEqual(d1_y.shape, (2500, ))


    def test_subsample_returns_nonoverlapping_sets(self):
        (d1_X, d1_y), (d2_X, d2_y) = utils.subsample(
                X,
                Y,
                2500,
                2,
                is_disjoint=True
                )
        total_samples = np.concatenate((d1_X, d2_X), axis=0)

        def check_values(d1_X, d2_X):
            is_unique = True
            for sample in d1_X:
                if sample in d2_X:
                    break
                    is_unique = False
            return is_unique

        self.assertEqual(check_values(d1_X, d2_X),  True)

    def test_subsample_returns_overlapping_sets_if_not_enough_samples(self):
        subsamples = utils.subsample(X[:3000], Y[:3000], 2500, 2, is_disjoint=False)
        (d1_X, d1_y), (d2_X, d2_y) = subsamples[0], subsamples[1]

        self.assertEqual(d1_X.shape, (2500, X.shape[1]))
        self.assertEqual(d1_y.shape, (2500, ))

        total_samples = np.concatenate((d1_X, d2_X), axis=0)
        unique_values = len(set(map(tuple, total_samples)))

        self.assertEqual(unique_values <= 3000, True)

if __name__ == '__main__':
    unittest.main()
