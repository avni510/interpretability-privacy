import numpy as np
import pandas as pd
import torch
import model_info as model_info
import os
from zipfile import ZipFile

def subsample(X, y, subsample_size, num_sets, is_disjoint=True):
    total_sample = subsample_size * num_sets
    indices = np.random.choice(
            X.shape[0],
            total_sample,
            replace = not is_disjoint
            )
    np.random.shuffle(indices)

    random_X = X[indices]
    random_y = y[indices]

    X_sets = np.split(random_X, num_sets)
    y_sets = np.split(random_y, num_sets)

    return list(zip(X_sets, y_sets))

def save_model_infos(experiment_dir, model_infos, new_dir_name):
    for info in model_infos:
        model_name = info['model_name']
        with ZipFile(experiment_dir + new_dir_name + '/' + model_name + ".zip", 'w') as zipObj:
            data_keys = list(info.keys())[1:]
            for key in data_keys:
                value = info[key]
                file_name = model_name + "_" + key + ".pt"
                path = experiment_dir + new_dir_name + '/' + file_name
                if torch.is_tensor(value):
                    value = value.to('cpu')
                    torch.save(value, path)
                    zipObj.write(path, arcname=file_name)
                    os.remove(path)
                else:
                    if value:
                        torch.save(value, path)
                        zipObj.write(path, arcname=file_name)
                        os.remove(path)
        zipObj.close()

def load_files(dir_name, info, i):
    data_keys = list(info.keys())[1:]
    for key in data_keys:
        path = dir_name + '/model_' + str(i) + "_" + key + '.pt'
        if os.path.exists(path):
            value = torch.load(path, map_location=torch.device('cpu'))
            info[key] = value
            os.remove(path)
    return info

def load_model_infos(experiment_dir, dir_name, num_models):
    model_infos = []
    for i in range(0, num_models):
        info = model_info.init_model_info()
        info['model_name'] = "model_" + str(i)

        zipfile_path = experiment_dir + dir_name + '/model_' + str(i) + '.zip'
        if os.path.exists(zipfile_path):
            with ZipFile(zipfile_path, 'r') as zipObj:
                zipObj.extractall(experiment_dir + dir_name)

        info = load_files(experiment_dir + dir_name, info, i)
        model_infos.append(info)

    return model_infos

def create_new_dir(experiment_dir, name):
    if not os.path.exists(experiment_dir + name):
        os.mkdir(experiment_dir + name)
