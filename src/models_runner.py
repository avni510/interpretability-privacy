import numpy as np
import torch
from adult_network import AdultNetwork
import train_network as train_network
import dataloader as dataloader
import os
import model_info
import utils as utils
import attack as attack

def create_models(in_features, num=4):
    layer_sizes = [20, 20, 20, 20, 2]
    models = []
    for i in range(0, 4):
        model = AdultNetwork(in_features, layer_sizes)
        models.append(model)
    return models

def train_models(models, dataloaders, model_infos, model_hyperparams, log_tensorboard=False):
    for idx, model in enumerate(models):
        train_loader = dataloaders[idx][0]
        info = model_infos[idx]
        model, train_losses, train_accuracies = train_network.train(
                model,
                train_loader,
                idx,
                model_hyperparams,
                log_tensorboard
                )
        info['training_losses'] = train_losses
        info['training_accuracies'] = train_accuracies
        info['model_params'] = model.state_dict()

    return model_infos

def test_models(models, dataloaders, model_infos):
    losses_accuracies = []
    for idx, model in enumerate(models):
        test_loader = dataloaders[idx][1]
        info = model_infos[idx]
        test_loss, test_accuracy = train_network.test(
                model,
                info['model_params'],
                idx,
                test_loader
                )
        info['test_loss'] = test_loss
        info['test_accuracy'] = test_accuracy

    return model_infos

def get_model_attributions(models, dataloaders, model_infos):
    model_attributions = []

    for idx, (model, dataloader) in enumerate(list(zip(models, dataloaders))):
        train_loader = dataloader[0]
        test_loader = dataloader[1]
        info = model_infos[idx]

        info['train_attributions'] =  train_network.get_attributions(
                model,
                info['model_params'],
                idx,
                train_loader
                )
        info['test_attributions'] = train_network.get_attributions(
                model,
                info['model_params'],
                idx,
                test_loader
                )

    return model_infos

def populate_train_test_sets(subsets, model_infos):
    for i, info in enumerate(model_infos):
        train = subsets[i]
        test = subsets[i + 1] if i + 1 < len(subsets) else subsets[0]
        train_normalize, test_normalize = dataloader.normalize(train, test)

        info['train_set'] = train_normalize
        info['test_set'] = test_normalize

    return model_infos

def init_model_infos(sets):
    model_infos = []
    for i in range(0, sets):
        m_info = model_info.init_model_info()
        m_info['model_name'] = 'model_' + str(i)

        model_infos.append(m_info)
    return model_infos

def split_data(X, Y, subsample_size, sets):
    model_infos = init_model_infos(sets)
    subsets = utils.subsample(X, Y, subsample_size, sets)
    model_infos = populate_train_test_sets(subsets, model_infos)
    return model_infos

def execute_training(model_infos, models, model_hyperparams):
    models_train_test = [(info['train_set'], info['test_set']) for info in model_infos]
    dataloaders = dataloader.get_loaders(models_train_test, model_hyperparams['batch_size'])

    model_infos = train_models(
            models,
            dataloaders,
            model_infos,
            model_hyperparams,
            log_tensorboard=False
            )

    return model_infos, dataloaders

def execute(
        subsample_size,
        sets,
        experiment_dir,
        num_iter,
        base_hyperparams,
        attack_hyperparams
        ):
    X, Y = dataloader.process_data()
    for i in range(0, num_iter):
        new_dir_name = '/iter_' + str(i)
        utils.create_new_dir(experiment_dir, new_dir_name)

        model_infos = split_data(X, Y, subsample_size, sets)
        utils.save_model_infos(experiment_dir, model_infos, new_dir_name)

        in_features = X.shape[1]
        models = create_models(in_features)

        print("--------------------------------")
        print("STARTING TRAINING")

        model_infos, dataloaders = execute_training(model_infos, models, base_hyperparams)
        utils.save_model_infos(experiment_dir, model_infos, new_dir_name)

        print("--------------------------------")
        print("STARTING TESTING")

        model_infos = test_models(models, dataloaders, model_infos)
        utils.save_model_infos(experiment_dir, model_infos, new_dir_name)

        print("--------------------------------")
        print("STARTING ATTRIBUTIONS")

        model_infos = get_model_attributions(models, dataloaders, model_infos)
        utils.save_model_infos(experiment_dir, model_infos, new_dir_name)

        print("--------------------------------")
        print("STARTING ATTACK")

        save_model_fn = lambda model_infos: utils.save_model_infos(experiment_dir, model_infos, new_dir_name)
        model_infos = attack.run(i, in_features, save_model_fn, attack_hyperparams, model_infos)

        print("--------------------------------")
        print("FINISHED")
