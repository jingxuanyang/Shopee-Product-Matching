import sys
sys.path.insert(0,'../input/shopee-competition-utils')

from config import CFG
from run_test import run_bert_test

# choose which cuda to load model on
CFG.DEVICE = 'cuda:1'
CFG.BATCH_SIZE = 16

# choose which model with what hyperparameters to use
CFG.BERT_MODEL_NAME = CFG.BERT_MODEL_NAMES[3]
CFG.MARGIN = CFG.BERT_MARGINS[3]
CFG.MODEL_PATH_BERT = f"{CFG.BERT_MODEL_NAME.rsplit('/', 1)[-1]}_epoch8-bs16x1_margin_{CFG.MARGIN}.pt"

# start inference
run_bert_test()
