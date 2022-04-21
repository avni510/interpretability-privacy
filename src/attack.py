import numpy as np
import itertools
import dataloader as dataloader
import os
from attack_network import AttackNetwork
import train_network as train_network

ATTACK_BATCH_SIZE = 8
ROOT_DIR = os.path.dirname(os.getcwd())
EXPERIMENT_DIR = ROOT_DIR + "/experiments/adult"

"""
The surrogate models are all the models that are not
the target model
"""
def get_surrogate_models(target_model_idx, model_infos):
    idxs = [i for i in range(0, len(model_infos))]
    idxs.remove(target_model_idx)
    surrogate_models = [model_infos[idx] for idx in idxs]
    return surrogate_models

"""
constructs a training sample that is the explanation vector (attribution),
target class the attribution is for, and if it belongs in the surrogate model's
training set or not.
is_in_train represents if it is in the surrogate model's training dataset
"""
def construct_class_attribution(attr_tuple, is_train):
    attribution = attr_tuple[1]
    target_class = attr_tuple[2]
    is_in_train = 1 if is_train else 0
    return {'attribution': attribution[0], 'class': target_class, 'is_in_train': is_in_train}

"""
For each surrogate model, we take the training data of that surrogate model and
those are used in the attack training dataset as 'in' the surrogate model training set. The
surrogate model test data is used in the attack training dataset as 'out'
of the surrogate model training set
"""
def create_attack_train_dataset(surrogate_models):
    attack_train_dataset = []
    for surrogate_info in surrogate_models:
        in_attributions = list(map(
            lambda attr_tuple: construct_class_attribution(attr_tuple, is_train=True),
            surrogate_info['train_attributions']
            ))
        out_attributions = list(map(
            lambda attr_tuple: construct_class_attribution(attr_tuple, is_train=False),
            surrogate_info['test_attributions']
            ))

        attack_train_dataset += in_attributions
        attack_train_dataset += out_attributions

    return attack_train_dataset

"""
We are testing if we are able to attack and correct guess if a sample is in the target
model's training dataset. This is what the test set is made of.
"""
def create_attack_test_dataset(target_model):
    in_attributions = list(map(
        lambda attr_tuple: construct_class_attribution(attr_tuple, is_train=True),
        target_model['train_attributions']
        ))
    out_attributions = list(map(
        lambda attr_tuple: construct_class_attribution(attr_tuple, is_train=False),
        target_model['test_attributions']
        ))
    attack_test_dataset = in_attributions + out_attributions
    return attack_test_dataset

def groupby_class(attack_dataset):
    attr_by_class = {}
    sortkeyfn = lambda attr_map: attr_map['class']
    for key, valuesiter in itertools.groupby(attack_dataset, key = sortkeyfn):
        attr_by_class[key] = list((v['attribution'], v['is_in_train']) for v in valuesiter)
    return attr_by_class

"""
attr_by_class is a dictionary of target class as key and value is a list of
{'attribution': val, 'class': val, 'is_in_train': val}. This function
regroups the data so it is a dictionary of the target class as a key
and the value is a list of attributions (which are X) and if that
attribution is in the model's training dataset or not
"""
def reorg_data(attr_by_class):
    data_by_class = {}
    for key, list_attrs in attr_by_class.items():
        X = [attr[0] for attr in list_attrs]
        y = [attr[1] for attr in list_attrs]
        data_by_class[key] = (X, y)

    return data_by_class

def normalize_data(class_by_train_test):
    scaled_class_by_train_test = {}
    for key, data in class_by_train_test.items():
        train, test = data
        train_scaled, test_scaled = dataloader.normalize(train, test)
        scaled_class_by_train_test[key] = (train_scaled, test_scaled)
    return scaled_class_by_train_test


def group_train_test(data_by_class_train, data_by_class_test):
    class_by_train_test = {}
    for key in data_by_class_train:
        train = data_by_class_train[key]
        test = data_by_class_test[key]
        class_by_train_test[key] = (train, test)
    return class_by_train_test

def convert_to_dataloaders(class_by_train_test):
    class_by_dataloaders = {}
    for key, data in class_by_train_test.items():
        train, test = data
        train_loader, test_loader = dataloader.get_loaders([(train, test)], ATTACK_BATCH_SIZE)[0]
        class_by_dataloaders[key] = (train_loader, test_loader)
    return class_by_dataloaders

"""
for the attack training dataset, asserts the samples in the surrogate models
training sets and out of surrogate models training set lie between 45% - 55%,
and for the attack test dataset, asserts the samples in the target model's
training set and out of target model's training set lies between 45% - 55%
this ensures that the attack model should do better than randomly guessing (50%)
if the sample is in the target model's training dataset
"""
def assert_ratio_in_members(data_by_class):
    # y values from class 0
    y_0 = np.array(data_by_class[0][1])
    # number of y values for class 0 that are out of the training set
    values_out = np.sum(y_0 == 0)
    # number of y values for class 0 that are in the training set
    values_in = np.sum(y_0 == 1)

    assert values_out/len(y_0) >= .45 and values_out/len(y_0) <= .55
    assert values_in/len(y_0) >= .45 and values_in/len(y_0) <= .55

    # y values from class 1
    y_1 = np.array(data_by_class[1][1])
    # number of y values for class 1 that are out of the training set
    values_out = np.sum(y_1 == 0)
    # number of y values for class 1 that are in the training set
    values_in = np.sum(y_1 == 1)

    assert values_out/len(y_1) >= .45 and values_out/len(y_1) <= .55
    assert values_in/len(y_1) >= .45 and values_in/len(y_1) <= .55


def create_dataloaders_by_class(attack_train_dataset, attack_test_dataset):
    attack_train_dataset = sorted(attack_train_dataset, key = lambda attr_map: attr_map['class'])
    attack_test_dataset = sorted(attack_test_dataset, key = lambda attr_map: attr_map['class'])

    attr_by_class_train = groupby_class(attack_train_dataset)
    attr_by_class_test = groupby_class(attack_test_dataset)

    data_by_class_train = reorg_data(attr_by_class_train)
    assert_ratio_in_members(data_by_class_train)
    data_by_class_test = reorg_data(attr_by_class_test)
    assert_ratio_in_members(data_by_class_test)

    class_by_train_test = group_train_test(data_by_class_train, data_by_class_test)
    class_by_train_test = normalize_data(class_by_train_test)

    return convert_to_dataloaders(class_by_train_test)

def load_model_infos(iter_num):
    model_infos = []
    for j in range(0, 4):
        path = EXPERIMENT_DIR + '/iter_' + str(iter_num) + '/model_' + str(j) + '.npy'
        model_info = np.load(path, allow_pickle=True)
        model_infos.append(model_info.item())
    return model_infos

def attack_model_training(class_by_dataloaders, in_features):
    attack_train_losses = {}
    attack_train_accuracies = {}
    attack_model_params = {}
    for key, dataloaders in class_by_dataloaders.items():
        model = AttackNetwork(in_features)
        train_loader = dataloaders[0]
        model, train_losses, train_accuracies = train_network.train_attack_model(
                model,
                train_loader,
                len(train_loader.dataset),
                log_tensorboard=True
                )
        attack_train_losses[key] = train_losses
        attack_train_accuracies[key] = train_accuracies
        attack_model_params[key] = model.state_dict()

    return attack_model_params, attack_train_losses, attack_train_accuracies

def attack_model_testing(target_model, idx, class_by_dataloaders, in_features):
    attack_test_loss = {}
    attack_test_accuracy = {}
    for key, model_params in target_model['attack_model_params'].items():
        model = AttackNetwork(in_features)
        model_params = model_params
        test_loader = class_by_dataloaders[key][1]
        test_loss, test_accuracy = train_network.test(
                model,
                model_params,
                idx,
                test_loader,
                len(test_loader.dataset),
                )
        attack_test_loss[key] = test_loss
        attack_test_accuracy[key] = test_accuracy

    return attack_test_loss, attack_test_accuracy


def run(iter_val, in_features, save_model_fn, model_infos=[]):
    if not model_infos:
        model_infos = load_model_infos(iter_val)

    for idx, info in enumerate(model_infos):
        target_model = info
        surrogate_models = get_surrogate_models(idx, model_infos)

        attack_train_dataset = create_attack_train_dataset(surrogate_models)
        attack_test_dataset = create_attack_test_dataset(target_model)

        class_by_dataloaders = create_dataloaders_by_class(
                attack_train_dataset,
                attack_test_dataset
                )

        attack_model_params, attack_train_losses, attack_train_accuracies = attack_model_training(
                class_by_dataloaders,
                in_features
                )

        target_model['attack_train_losses'] = attack_train_losses
        target_model['attack_train_accuracies'] = attack_train_accuracies
        target_model['attack_model_params'] = attack_model_params

        save_model_fn(model_infos)

        attack_test_loss, attack_test_accuracy = attack_model_testing(
                target_model,
                idx,
                class_by_dataloaders,
                in_features
                )

        target_model['attack_test_loss'] = attack_test_loss
        target_model['attack_test_accuracy'] = attack_test_accuracy

        save_model_fn(model_infos)

    return model_infos
