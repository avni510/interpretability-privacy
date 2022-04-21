import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import torch
import utils as utils
import os
from sklearn import preprocessing

def process_data():
    path = os.path.join(os.path.dirname(__file__), "../data/adult.csv")
    df = pd.read_csv(path)
    updated_df = pd.get_dummies(df).drop_duplicates()

    # The paper describes they were able to get to 104 features.
    # print(len(updated_df.columns))

    dataset = updated_df.to_numpy()

    # remove the columns for income_<=50K and income_>50K
    # y values are income_>50K
    Y = dataset[:, -1]
    X = np.delete(dataset, -1, axis = 1)
    X = np.delete(X, -1, axis = 1)

    return X, Y

def get_data_columns():
    path = os.path.join(os.path.dirname(__file__), "../data/adult.csv")
    df = pd.read_csv(path)
    updated_df = pd.get_dummies(df).drop_duplicates()
    return updated_df.columns


def normalize(train, test):
    train_X, train_y = train
    test_X, test_y = test

    train_X = np.array([sample.numpy() for sample in train_X])
    test_X = np.array([sample.numpy() for sample in test_X])

    scaler = preprocessing.StandardScaler().fit(train_X)
    train_X_scaled = scaler.transform(train_X)
    test_X_scaled = scaler.transform(test_X)

    return (train_X_scaled, train_y), (test_X_scaled, test_y)

def get_loaders(model_sets, batch_size):
    datasets = list(map(
        lambda data: (
            TensorDataset(torch.tensor(data[0][0]), torch.tensor(data[0][1])),
            TensorDataset(torch.tensor(data[1][0]), torch.tensor(data[1][1]))
            ),
        model_sets
        ))

    return list(map(
        lambda d: (
            DataLoader(dataset=d[0], batch_size=batch_size, shuffle=True),
            DataLoader(dataset=d[1], batch_size=batch_size, shuffle=True)
            ),
        datasets
        ))
