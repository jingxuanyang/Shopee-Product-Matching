import sys
sys.path.insert(0,'../input/shopee-competition-utils')

from sklearn.externals import joblib
from config import CFG
from run_test import run_bert_test_all
from run_training import run_bert_training

# choose which cuda to train model on
CFG.DEVICE = 'cuda:0'
CFG.BATCH_SIZE = 16

for i in range(len(CFG.BERT_MODEL_NAMES)):
    CFG.BERT_MODEL_NAME = CFG.BERT_MODEL_NAMES[i]
    for k in range(len(CFG.BERT_MARGINS)):
        CFG.MARGIN = CFG.BERT_MARGINS[k]
        CFG.MODEL_PATH_BERT = f"{CFG.BERT_MODEL_NAME.rsplit('/', 1)[-1]}_epoch8-bs16x1_margin_{CFG.MARGIN}.pt"
        run_bert_training()

CFG.USE_EMBEDDING = False # set `False` if use model to compute the embeddings
test_bert_result_dict = run_bert_test_all()

joblib.dump(test_bert_result_dict, CFG.RESULTS_SAVE_PATH + 'test_bert_result_dict.pkl')
test_bert_result_dict

print(joblib.load(CFG.RESULTS_SAVE_PATH + 'test_bert_result_dict.pkl'))
