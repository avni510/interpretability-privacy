#!/usr/bin/env python

import models_runner as models_runner
import yaml
import os

with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

SUBSAMPLE_SIZE = config['subsample_size']
SETS = config['sets']

ROOT_DIR = os.path.dirname(os.getcwd())
EXPERIMENT_DIR = ROOT_DIR + config['experiment_dir']
NUM_ITER = config['num_runs']
#
# BASE_BATCH_SIZE = config['base']['batch_size']
# BASE_EPOCHS = config['base']['epochs']
# BASE_LR = config['base']['lr']
# BASE_LR_DECAY = config['base']['lr_decay']
#
# ATTACK_BATCH_SIZE = config['attack']['batch_size']
# ATTACK_EPOCHS = config['attack']['epochs']
# ATTACK_LR = config['attack']['lr']
# ATTACK_LR_DECAY = config['attack'['lr_decay']

BASE = config['base_hyperparams']
ATTACK = config['attack_hyperparams']

try:
    models_runner.execute(
            SUBSAMPLE_SIZE,
            SETS,
            EXPERIMENT_DIR,
            NUM_ITER,
            BASE,
            ATTACK
            )
except Exception as e:
    print(e)
