import sys
sys.path.insert(0,'../input/shopee-competition-utils')

from config import CFG
from run_training import run_training
from run_test import run_test

# choose which cuda to train model on
CFG.DEVICE = 'cuda:0'

# choose which model with what hyperparameters to train
CFG.LOSS_MODULE = CFG.LOSS_MODULES[1]
CFG.MODEL_NAME = CFG.MODEL_NAMES[4]
CFG.MARGIN = CFG.MARGINS[2]
CFG.MODEL_PATH = f'{CFG.MODEL_NAME}_{CFG.LOSS_MODULE}_face_epoch_8_bs_8_margin_{CFG.MARGIN}.pt'

# start training
run_training()

# CFG.USE_EMBEDDING = False # set true if use model to compute the embeddings
run_test()
