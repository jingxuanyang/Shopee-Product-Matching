import torch
import numpy as np
import pandas as pd
from config import CFG
from read_dataset import read_dataset
from get_neighbors import get_voting_nns, get_voting_neighbors, get_union_neighbors, get_voting_predictions
from search_threshold import search_voting_threshold, search_inb_threshold
from seed_everything import seed_everything
from shopee_text_model import ShopeeBertModel
from get_embeddings import get_bert_embeddings, get_tfidf_embeddings
from blend_neighborhood import blend_neighborhood
from criterion import precision_score, recall_score, f1_score

def run_image_ensemble():
    """
    Note that model parameters for neil, min2 and inb are the same.
    """

    # resnet
    CFG.LOSS_MODULE = CFG.LOSS_MODULES[0]
    CFG.MODEL_NAME = CFG.MODEL_NAMES[0]
    CFG.MARGIN = CFG.MARGINS[1]
    CFG.MODEL_PATH = f'{CFG.MODEL_NAME}_{CFG.LOSS_MODULE}_face_epoch_8_bs_8_margin_{CFG.MARGIN}.pt'

    TEST_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_test_embed.csv'
    print(f'Loading {TEST_EMBEDDING_PATH} ...')
    test_embeddings_1 = np.loadtxt(TEST_EMBEDDING_PATH, delimiter=',')
    VALID_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_valid_embed.csv'
    print(f'Loading {VALID_EMBEDDING_PATH} ...')
    valid_embeddings_1 = np.loadtxt(VALID_EMBEDDING_PATH, delimiter=',')

    # resnext
    CFG.LOSS_MODULE = CFG.LOSS_MODULES[0]
    CFG.MODEL_NAME = CFG.MODEL_NAMES[1]
    CFG.MARGIN = CFG.MARGINS[3]
    CFG.MODEL_PATH = f'{CFG.MODEL_NAME}_{CFG.LOSS_MODULE}_face_epoch_8_bs_8_margin_{CFG.MARGIN}.pt'

    TEST_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_test_embed.csv'
    print(f'Loading {TEST_EMBEDDING_PATH} ...')
    test_embeddings_2 = np.loadtxt(TEST_EMBEDDING_PATH, delimiter=',')
    VALID_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_valid_embed.csv'
    print(f'Loading {VALID_EMBEDDING_PATH} ...')
    valid_embeddings_2 = np.loadtxt(VALID_EMBEDDING_PATH, delimiter=',')

    # densenet
    CFG.LOSS_MODULE = CFG.LOSS_MODULES[0]
    CFG.MODEL_NAME = CFG.MODEL_NAMES[2]
    CFG.MARGIN = CFG.MARGINS[4]
    CFG.MODEL_PATH = f'{CFG.MODEL_NAME}_{CFG.LOSS_MODULE}_face_epoch_8_bs_8_margin_{CFG.MARGIN}.pt'

    TEST_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_test_embed.csv'
    print(f'Loading {TEST_EMBEDDING_PATH} ...')
    test_embeddings_3 = np.loadtxt(TEST_EMBEDDING_PATH, delimiter=',')
    VALID_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_valid_embed.csv'
    print(f'Loading {VALID_EMBEDDING_PATH} ...')
    valid_embeddings_3 = np.loadtxt(VALID_EMBEDDING_PATH, delimiter=',')

    # efficientnet
    CFG.LOSS_MODULE = CFG.LOSS_MODULES[1]
    CFG.MODEL_NAME = CFG.MODEL_NAMES[3]
    CFG.MARGIN = CFG.MARGINS[0]
    CFG.MODEL_PATH = f'{CFG.MODEL_NAME}_{CFG.LOSS_MODULE}_face_epoch_8_bs_8_margin_{CFG.MARGIN}.pt'

    TEST_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_test_embed.csv'
    print(f'Loading {TEST_EMBEDDING_PATH} ...')
    test_embeddings_4 = np.loadtxt(TEST_EMBEDDING_PATH, delimiter=',')
    VALID_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_valid_embed.csv'
    print(f'Loading {VALID_EMBEDDING_PATH} ...')
    valid_embeddings_4 = np.loadtxt(VALID_EMBEDDING_PATH, delimiter=',')

    # nfnet
    CFG.LOSS_MODULE = CFG.LOSS_MODULES[0]
    CFG.MODEL_NAME = CFG.MODEL_NAMES[4]
    CFG.MARGIN = CFG.MARGINS[4]
    CFG.MODEL_PATH = f'{CFG.MODEL_NAME}_{CFG.LOSS_MODULE}_face_epoch_8_bs_8_margin_{CFG.MARGIN}.pt'

    TEST_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_test_embed.csv'
    print(f'Loading {TEST_EMBEDDING_PATH} ...')
    test_embeddings_5 = np.loadtxt(TEST_EMBEDDING_PATH, delimiter=',')
    VALID_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_valid_embed.csv'
    print(f'Loading {VALID_EMBEDDING_PATH} ...')
    valid_embeddings_5 = np.loadtxt(VALID_EMBEDDING_PATH, delimiter=',')

    # read dataset
    _, valid_df, test_df = read_dataset()

    # get voting of valid embeddings
    valid_embeddings_dict = {'emb_0':valid_embeddings_1, 
                             'emb_1':valid_embeddings_2,
                             'emb_2':valid_embeddings_3,
                             'emb_3':valid_embeddings_4,
                             'emb_4':valid_embeddings_5}
    distances, indices = get_voting_nns(valid_embeddings_dict)

    # search best thresholds
    search_voting_threshold(valid_df, distances, indices)

    # get voting of test embeddings
    test_embeddings_dict = {'emb_0':test_embeddings_1, 
                            'emb_1':test_embeddings_2,
                            'emb_2':test_embeddings_3,
                            'emb_3':test_embeddings_4,
                            'emb_4':test_embeddings_5}
    distances, indices = get_voting_nns(test_embeddings_dict)

    result_list = [[0 for i in range(3)] for j in range(2)]
    # use obtained thresholds to get test results
    test_df = get_voting_neighbors(test_df, distances, indices, threshold = CFG.BEST_THRESHOLD, min2 = False)
    test_f1 = test_df.f1.mean()
    test_recall = test_df.recall.mean()
    test_precision = test_df.precision.mean()
    result_list[0][0] = test_f1
    result_list[0][1] = test_recall
    result_list[0][2] = test_precision
    print(f'f1 score after voting = {test_f1}, recall = {test_recall}, precision = {test_precision}')

    # use obtained thresholds to get test results after min2
    test_df = get_voting_neighbors(test_df, distances, indices, threshold = CFG.BEST_THRESHOLD_MIN2, min2 = True)
    test_f1 = test_df.f1.mean()
    test_recall = test_df.recall.mean()
    test_precision = test_df.precision.mean()
    result_list[1][0] = test_f1
    result_list[1][1] = test_recall
    result_list[1][2] = test_precision
    print(f'f1 score after min2 and voting = {test_f1}, recall = {test_recall}, precision = {test_precision}')

    result_df = pd.DataFrame(result_list, columns=['f1','recall','precision'])

    return result_df

def run_text_ensemble():
    """
    """
    seed_everything(CFG.SEED_BERT)
    _, valid_df, test_df = read_dataset()
    valid_embeddings_dict = {}
    test_embeddings_dict = {}
    for i in range(len(CFG.BERT_MODEL_NAMES)):
        CFG.BERT_MODEL_NAME = CFG.BERT_MODEL_NAMES[i]
        CFG.MARGIN = CFG.BEST_BERT_MARGINS[0][i]
        model = ShopeeBertModel(
            model_name = CFG.BERT_MODEL_NAME,
            margin = CFG.MARGIN
        )
        CFG.MODEL_PATH_BERT = f"{CFG.BERT_MODEL_NAME.rsplit('/', 1)[-1]}_epoch8-bs16x1_margin_{CFG.MARGIN}.pt"
        SAVE_MODEL_PATH = CFG.TEXT_MODEL_PATH_PREFIX + CFG.MODEL_PATH_BERT
        print(CFG.MODEL_PATH_BERT)
        model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=CFG.DEVICE))
        valid_embeddings_dict[f'emb_{i}'] = get_bert_embeddings(valid_df, 'title', model)
        test_embeddings_dict[f'emb_{i}'] = get_bert_embeddings(test_df, 'title', model)

    distances, indices = get_voting_nns(valid_embeddings_dict)

    # search best thresholds
    search_voting_threshold(valid_df, distances, indices)
    distances, indices = get_voting_nns(test_embeddings_dict)

    result_list = [[0 for i in range(3)] for j in range(2)]
    # use obtained thresholds to get test results
    test_df = get_voting_neighbors(test_df, distances, indices, threshold = CFG.BEST_THRESHOLD, min2 = False)
    test_f1 = test_df.f1.mean()
    test_recall = test_df.recall.mean()
    test_precision = test_df.precision.mean()
    result_list[0][0] = test_f1
    result_list[0][1] = test_recall
    result_list[0][2] = test_precision
    print(f'f1 score after voting = {test_f1}, recall = {test_recall}, precision = {test_precision}')

    # use obtained thresholds to get test results after min2
    test_df = get_voting_neighbors(test_df, distances, indices, threshold = CFG.BEST_THRESHOLD_MIN2, min2 = True)
    test_f1 = test_df.f1.mean()
    test_recall = test_df.recall.mean()
    test_precision = test_df.precision.mean()
    result_list[1][0] = test_f1
    result_list[1][1] = test_recall
    result_list[1][2] = test_precision
    print(f'f1 score after min2 and voting = {test_f1}, recall = {test_recall}, precision = {test_precision}')

    result_df = pd.DataFrame(result_list, columns=['f1','recall','precision'])

    return result_df

def run_nfnet_sbert_ensemble():
    seed_everything(CFG.SEED_BERT)
    _, valid_df, test_df = read_dataset()

    # nfnet
    CFG.LOSS_MODULE = CFG.LOSS_MODULES[0]
    CFG.MODEL_NAME = CFG.MODEL_NAMES[4]
    CFG.MARGIN = CFG.MARGINS[4]
    CFG.MODEL_PATH = f'{CFG.MODEL_NAME}_{CFG.LOSS_MODULE}_face_epoch_8_bs_8_margin_{CFG.MARGIN}.pt'

    TEST_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_test_embed.csv'
    print(f'Loading {TEST_EMBEDDING_PATH} ...')
    image_test_embeddings = np.loadtxt(TEST_EMBEDDING_PATH, delimiter=',')
    VALID_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_valid_embed.csv'
    print(f'Loading {VALID_EMBEDDING_PATH} ...')
    image_valid_embeddings = np.loadtxt(VALID_EMBEDDING_PATH, delimiter=',')

    # paraphrase-xlm-r-multilingual-v1
    CFG.BERT_MODEL_NAME = CFG.BERT_MODEL_NAMES[3]
    CFG.MARGIN = CFG.BERT_MARGINS[3]
    model = ShopeeBertModel(
        model_name = CFG.BERT_MODEL_NAME,
        margin = CFG.MARGIN
    )
    CFG.MODEL_PATH_BERT = f"{CFG.BERT_MODEL_NAME.rsplit('/', 1)[-1]}_epoch8-bs16x1_margin_{CFG.MARGIN}.pt"
    SAVE_MODEL_PATH = CFG.TEXT_MODEL_PATH_PREFIX + CFG.MODEL_PATH_BERT
    print(CFG.MODEL_PATH_BERT)
    model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=CFG.DEVICE))
    text_valid_embeddings = get_bert_embeddings(valid_df, 'title', model)
    text_test_embeddings = get_bert_embeddings(test_df, 'title', model)

    # get voting of valid embeddings
    valid_embeddings_dict = {'emb_0':image_valid_embeddings,
                             'emb_1':text_valid_embeddings}
    distances, indices = get_voting_nns(valid_embeddings_dict)

    # search best thresholds
    search_voting_threshold(valid_df, distances, indices, upper=60)

    # get voting of test embeddings
    test_embeddings_dict = {'emb_0':image_test_embeddings,
                            'emb_1':text_test_embeddings}
    distances, indices = get_voting_nns(test_embeddings_dict)

    result_list = [[0 for i in range(3)] for j in range(2)]
    # use obtained thresholds to get test results
    test_df = get_voting_neighbors(test_df, distances, indices, threshold = CFG.BEST_THRESHOLD, min2 = False)
    test_f1 = test_df.f1.mean()
    test_recall = test_df.recall.mean()
    test_precision = test_df.precision.mean()
    result_list[0][0] = test_f1
    result_list[0][1] = test_recall
    result_list[0][2] = test_precision
    print(f'f1 score after voting = {test_f1}, recall = {test_recall}, precision = {test_precision}')

    # use obtained thresholds to get test results after min2
    test_df = get_voting_neighbors(test_df, distances, indices, threshold = CFG.BEST_THRESHOLD_MIN2, min2 = True)
    test_f1 = test_df.f1.mean()
    test_recall = test_df.recall.mean()
    test_precision = test_df.precision.mean()
    result_list[1][0] = test_f1
    result_list[1][1] = test_recall
    result_list[1][2] = test_precision
    print(f'f1 score after min2 and voting = {test_f1}, recall = {test_recall}, precision = {test_precision}')

    result_df = pd.DataFrame(result_list, columns=['f1','recall','precision'])

    return result_df

def run_tfidf_resnet_ensemble():
    seed_everything(CFG.SEED)
    _, valid_df, test_df = read_dataset()

    # tf-idf
    text_valid_embeddings = get_tfidf_embeddings(valid_df)
    text_test_embeddings = get_tfidf_embeddings(test_df)

    # resnet
    CFG.LOSS_MODULE = CFG.LOSS_MODULES[0]
    CFG.MODEL_NAME = CFG.MODEL_NAMES[0]
    CFG.MARGIN = CFG.MARGINS[1]
    CFG.MODEL_PATH = f'{CFG.MODEL_NAME}_{CFG.LOSS_MODULE}_face_epoch_8_bs_8_margin_{CFG.MARGIN}.pt'

    TEST_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_test_embed.csv'
    print(f'Loading {TEST_EMBEDDING_PATH} ...')
    image_test_embeddings = np.loadtxt(TEST_EMBEDDING_PATH, delimiter=',')
    VALID_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_valid_embed.csv'
    print(f'Loading {VALID_EMBEDDING_PATH} ...')
    image_valid_embeddings = np.loadtxt(VALID_EMBEDDING_PATH, delimiter=',')

    # get voting of valid embeddings
    valid_embeddings_dict = {'emb_0':image_valid_embeddings,
                             'emb_1':text_valid_embeddings}
    distances, indices = get_voting_nns(valid_embeddings_dict)

    # search best thresholds
    search_voting_threshold(valid_df, distances, indices, upper=60)

    # get voting of test embeddings
    test_embeddings_dict = {'emb_0':image_test_embeddings,
                            'emb_1':text_test_embeddings}
    distances, indices = get_voting_nns(test_embeddings_dict)

    result_list = [[0 for i in range(3)] for j in range(2)]
    # use obtained thresholds to get test results
    test_df = get_voting_neighbors(test_df, distances, indices, threshold = CFG.BEST_THRESHOLD, min2 = False)
    test_f1 = test_df.f1.mean()
    test_recall = test_df.recall.mean()
    test_precision = test_df.precision.mean()
    result_list[0][0] = test_f1
    result_list[0][1] = test_recall
    result_list[0][2] = test_precision
    print(f'f1 score after voting = {test_f1}, recall = {test_recall}, precision = {test_precision}')

    # use obtained thresholds to get test results after min2
    test_df = get_voting_neighbors(test_df, distances, indices, threshold = CFG.BEST_THRESHOLD_MIN2, min2 = True)
    test_f1 = test_df.f1.mean()
    test_recall = test_df.recall.mean()
    test_precision = test_df.precision.mean()
    result_list[1][0] = test_f1
    result_list[1][1] = test_recall
    result_list[1][2] = test_precision
    print(f'f1 score after min2 and voting = {test_f1}, recall = {test_recall}, precision = {test_precision}')

    result_df = pd.DataFrame(result_list, columns=['f1','recall','precision'])

    return result_df

def combine_predictions(row):
    x = np.concatenate([row['image_predictions'], row['text_predictions']])
    return ' '.join(np.unique(x))

def run_tfidf_resnet_union():
    seed_everything(CFG.SEED)
    _, valid_df, test_df = read_dataset()
    result_list = [[0 for i in range(3)] for j in range(3)]

    # tf-idf
    text_valid_embeddings = get_tfidf_embeddings(valid_df)
    text_test_embeddings = get_tfidf_embeddings(test_df)

    # resnet
    CFG.LOSS_MODULE = CFG.LOSS_MODULES[0]
    CFG.MODEL_NAME = CFG.MODEL_NAMES[0]
    CFG.MARGIN = CFG.MARGINS[1]
    CFG.MODEL_PATH = f'{CFG.MODEL_NAME}_{CFG.LOSS_MODULE}_face_epoch_8_bs_8_margin_{CFG.MARGIN}.pt'

    TEST_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_test_embed.csv'
    print(f'Loading {TEST_EMBEDDING_PATH} ...')
    image_test_embeddings = np.loadtxt(TEST_EMBEDDING_PATH, delimiter=',')
    VALID_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_valid_embed.csv'
    print(f'Loading {VALID_EMBEDDING_PATH} ...')
    image_valid_embeddings = np.loadtxt(VALID_EMBEDDING_PATH, delimiter=',')

    # text predictions
    search_inb_threshold(valid_df,text_valid_embeddings,lower=40,upper=70)
    text_predictions = get_union_neighbors(test_df, text_test_embeddings, threshold=CFG.BEST_THRESHOLD)

    # text predictions, min2
    text_predictions_min2 = get_union_neighbors(test_df, text_test_embeddings, threshold=CFG.BEST_THRESHOLD_MIN2, min2=True)

    # text predictions, inb
    new_valid_emb = blend_neighborhood(valid_df,text_valid_embeddings)
    search_inb_threshold(valid_df,new_valid_emb)
    new_test_emb = blend_neighborhood(test_df,text_test_embeddings)
    text_predictions_inb = get_union_neighbors(test_df, new_test_emb, threshold=CFG.BEST_THRESHOLD_MIN2, min2 = True)

    # image predictions
    search_inb_threshold(valid_df,image_valid_embeddings)
    image_predictions = get_union_neighbors(test_df, image_test_embeddings, threshold=CFG.BEST_THRESHOLD)

    # image predictions, min2
    image_predictions_min2 = get_union_neighbors(test_df, image_test_embeddings, threshold=CFG.BEST_THRESHOLD_MIN2, min2=True)

    # image predictions, inb
    new_valid_emb = blend_neighborhood(valid_df,image_valid_embeddings)
    search_inb_threshold(valid_df,new_valid_emb)
    new_test_emb = blend_neighborhood(test_df,image_test_embeddings)
    image_predictions_inb = get_union_neighbors(test_df, new_test_emb, threshold=CFG.BEST_THRESHOLD_MIN2, min2 = True)

    test_df['image_predictions'] = image_predictions
    test_df['text_predictions'] = text_predictions
    test_df['pred_matches'] = test_df.apply(combine_predictions, axis = 1)
    test_df['f1'] = f1_score(test_df['matches'], test_df['pred_matches'])
    test_df['recall'] = recall_score(test_df['matches'], test_df['pred_matches'])
    test_df['precision'] = precision_score(test_df['matches'], test_df['pred_matches'])
    test_f1 = test_df.f1.mean()
    test_recall = test_df.recall.mean()
    test_precision = test_df.precision.mean()
    result_list[0][0] = test_f1
    result_list[0][1] = test_recall
    result_list[0][2] = test_precision
    print(f'Test f1 score = {test_f1}, recall = {test_recall}, precision = {test_precision}')

    # Min2
    test_df['image_predictions'] = image_predictions_min2
    test_df['text_predictions'] = text_predictions_min2
    test_df['pred_matches'] = test_df.apply(combine_predictions, axis = 1)
    test_df['f1'] = f1_score(test_df['matches'], test_df['pred_matches'])
    test_df['recall'] = recall_score(test_df['matches'], test_df['pred_matches'])
    test_df['precision'] = precision_score(test_df['matches'], test_df['pred_matches'])
    test_f1 = test_df.f1.mean()
    test_recall = test_df.recall.mean()
    test_precision = test_df.precision.mean()
    result_list[1][0] = test_f1
    result_list[1][1] = test_recall
    result_list[1][2] = test_precision
    print(f'Test f1 score after min2 = {test_f1}, recall = {test_recall}, precision = {test_precision}')

    # INB
    test_df['image_predictions'] = image_predictions_inb
    test_df['text_predictions'] = text_predictions_inb
    test_df['pred_matches'] = test_df.apply(combine_predictions, axis = 1)
    test_df['f1'] = f1_score(test_df['matches'], test_df['pred_matches'])
    test_df['recall'] = recall_score(test_df['matches'], test_df['pred_matches'])
    test_df['precision'] = precision_score(test_df['matches'], test_df['pred_matches'])
    test_f1 = test_df.f1.mean()
    test_recall = test_df.recall.mean()
    test_precision = test_df.precision.mean()
    result_list[2][0] = test_f1
    result_list[2][1] = test_recall
    result_list[2][2] = test_precision
    print(f'Test f1 score after INB = {test_f1}, recall = {test_recall}, precision = {test_precision}')

    result_df = pd.DataFrame(result_list,columns=['f1','recall','precision'])

    return result_df

def run_nfnet_sbert_union():
    seed_everything(CFG.SEED_BERT)
    _, valid_df, test_df = read_dataset()
    result_list = [[0 for i in range(3)] for j in range(3)]

    # nfnet
    CFG.LOSS_MODULE = CFG.LOSS_MODULES[0]
    CFG.MODEL_NAME = CFG.MODEL_NAMES[4]
    CFG.MARGIN = CFG.MARGINS[4]
    CFG.MODEL_PATH = f'{CFG.MODEL_NAME}_{CFG.LOSS_MODULE}_face_epoch_8_bs_8_margin_{CFG.MARGIN}.pt'

    TEST_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_test_embed.csv'
    print(f'Loading {TEST_EMBEDDING_PATH} ...')
    image_test_embeddings = np.loadtxt(TEST_EMBEDDING_PATH, delimiter=',')
    VALID_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_valid_embed.csv'
    print(f'Loading {VALID_EMBEDDING_PATH} ...')
    image_valid_embeddings = np.loadtxt(VALID_EMBEDDING_PATH, delimiter=',')

    # paraphrase-xlm-r-multilingual-v1
    CFG.BERT_MODEL_NAME = CFG.BERT_MODEL_NAMES[3]
    CFG.MARGIN = CFG.BERT_MARGINS[3]
    model = ShopeeBertModel(
        model_name = CFG.BERT_MODEL_NAME,
        margin = CFG.MARGIN
    )
    CFG.MODEL_PATH_BERT = f"{CFG.BERT_MODEL_NAME.rsplit('/', 1)[-1]}_epoch8-bs16x1_margin_{CFG.MARGIN}.pt"
    SAVE_MODEL_PATH = CFG.TEXT_MODEL_PATH_PREFIX + CFG.MODEL_PATH_BERT
    print(CFG.MODEL_PATH_BERT)
    model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=CFG.DEVICE))
    text_valid_embeddings = get_bert_embeddings(valid_df, 'title', model)
    text_test_embeddings = get_bert_embeddings(test_df, 'title', model)

    # text predictions
    search_inb_threshold(valid_df,text_valid_embeddings)
    text_predictions = get_union_neighbors(test_df, text_test_embeddings, threshold=CFG.BEST_THRESHOLD)

    # text predictions, min2
    text_predictions_min2 = get_union_neighbors(test_df, text_test_embeddings, threshold=CFG.BEST_THRESHOLD_MIN2, min2=True)

    # text predictions, inb
    new_valid_emb = blend_neighborhood(valid_df,text_valid_embeddings)
    search_inb_threshold(valid_df,new_valid_emb)
    new_test_emb = blend_neighborhood(test_df,text_test_embeddings)
    text_predictions_inb = get_union_neighbors(test_df, new_test_emb, threshold=CFG.BEST_THRESHOLD_MIN2, min2 = True)

    # image predictions
    search_inb_threshold(valid_df,image_valid_embeddings)
    image_predictions = get_union_neighbors(test_df, image_test_embeddings, threshold=CFG.BEST_THRESHOLD)

    # image predictions, min2
    image_predictions_min2 = get_union_neighbors(test_df, image_test_embeddings, threshold=CFG.BEST_THRESHOLD_MIN2, min2=True)

    # image predictions, inb
    new_valid_emb = blend_neighborhood(valid_df,image_valid_embeddings)
    search_inb_threshold(valid_df,new_valid_emb)
    new_test_emb = blend_neighborhood(test_df,image_test_embeddings)
    image_predictions_inb = get_union_neighbors(test_df, new_test_emb, threshold=CFG.BEST_THRESHOLD_MIN2, min2 = True)

    test_df['image_predictions'] = image_predictions
    test_df['text_predictions'] = text_predictions
    test_df['pred_matches'] = test_df.apply(combine_predictions, axis = 1)
    test_df['f1'] = f1_score(test_df['matches'], test_df['pred_matches'])
    test_df['recall'] = recall_score(test_df['matches'], test_df['pred_matches'])
    test_df['precision'] = precision_score(test_df['matches'], test_df['pred_matches'])
    test_f1 = test_df.f1.mean()
    test_recall = test_df.recall.mean()
    test_precision = test_df.precision.mean()
    result_list[0][0] = test_f1
    result_list[0][1] = test_recall
    result_list[0][2] = test_precision
    print(f'Test f1 score = {test_f1}, recall = {test_recall}, precision = {test_precision}')

    # Min2
    test_df['image_predictions'] = image_predictions_min2
    test_df['text_predictions'] = text_predictions_min2
    test_df['pred_matches'] = test_df.apply(combine_predictions, axis = 1)
    test_df['f1'] = f1_score(test_df['matches'], test_df['pred_matches'])
    test_df['recall'] = recall_score(test_df['matches'], test_df['pred_matches'])
    test_df['precision'] = precision_score(test_df['matches'], test_df['pred_matches'])
    test_f1 = test_df.f1.mean()
    test_recall = test_df.recall.mean()
    test_precision = test_df.precision.mean()
    result_list[1][0] = test_f1
    result_list[1][1] = test_recall
    result_list[1][2] = test_precision
    print(f'Test f1 score after min2 = {test_f1}, recall = {test_recall}, precision = {test_precision}')

    # INB
    test_df['image_predictions'] = image_predictions_inb
    test_df['text_predictions'] = text_predictions_inb
    test_df['pred_matches'] = test_df.apply(combine_predictions, axis = 1)
    test_df['f1'] = f1_score(test_df['matches'], test_df['pred_matches'])
    test_df['recall'] = recall_score(test_df['matches'], test_df['pred_matches'])
    test_df['precision'] = precision_score(test_df['matches'], test_df['pred_matches'])
    test_f1 = test_df.f1.mean()
    test_recall = test_df.recall.mean()
    test_precision = test_df.precision.mean()
    result_list[2][0] = test_f1
    result_list[2][1] = test_recall
    result_list[2][2] = test_precision
    print(f'Test f1 score after INB = {test_f1}, recall = {test_recall}, precision = {test_precision}')

    result_df = pd.DataFrame(result_list,columns=['f1','recall','precision'])

    return result_df

def run_text_image_union():
    # read dataset
    seed_everything(CFG.SEED_BERT)
    _, valid_df, test_df = read_dataset()

    ##################################### image ####################################
    # resnet
    CFG.LOSS_MODULE = CFG.LOSS_MODULES[0]
    CFG.MODEL_NAME = CFG.MODEL_NAMES[0]
    CFG.MARGIN = CFG.MARGINS[1]
    CFG.MODEL_PATH = f'{CFG.MODEL_NAME}_{CFG.LOSS_MODULE}_face_epoch_8_bs_8_margin_{CFG.MARGIN}.pt'

    TEST_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_test_embed.csv'
    print(f'Loading {TEST_EMBEDDING_PATH} ...')
    test_embeddings_1 = np.loadtxt(TEST_EMBEDDING_PATH, delimiter=',')
    VALID_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_valid_embed.csv'
    print(f'Loading {VALID_EMBEDDING_PATH} ...')
    valid_embeddings_1 = np.loadtxt(VALID_EMBEDDING_PATH, delimiter=',')

    # resnext
    CFG.LOSS_MODULE = CFG.LOSS_MODULES[0]
    CFG.MODEL_NAME = CFG.MODEL_NAMES[1]
    CFG.MARGIN = CFG.MARGINS[3]
    CFG.MODEL_PATH = f'{CFG.MODEL_NAME}_{CFG.LOSS_MODULE}_face_epoch_8_bs_8_margin_{CFG.MARGIN}.pt'

    TEST_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_test_embed.csv'
    print(f'Loading {TEST_EMBEDDING_PATH} ...')
    test_embeddings_2 = np.loadtxt(TEST_EMBEDDING_PATH, delimiter=',')
    VALID_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_valid_embed.csv'
    print(f'Loading {VALID_EMBEDDING_PATH} ...')
    valid_embeddings_2 = np.loadtxt(VALID_EMBEDDING_PATH, delimiter=',')

    # densenet
    CFG.LOSS_MODULE = CFG.LOSS_MODULES[0]
    CFG.MODEL_NAME = CFG.MODEL_NAMES[2]
    CFG.MARGIN = CFG.MARGINS[4]
    CFG.MODEL_PATH = f'{CFG.MODEL_NAME}_{CFG.LOSS_MODULE}_face_epoch_8_bs_8_margin_{CFG.MARGIN}.pt'

    TEST_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_test_embed.csv'
    print(f'Loading {TEST_EMBEDDING_PATH} ...')
    test_embeddings_3 = np.loadtxt(TEST_EMBEDDING_PATH, delimiter=',')
    VALID_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_valid_embed.csv'
    print(f'Loading {VALID_EMBEDDING_PATH} ...')
    valid_embeddings_3 = np.loadtxt(VALID_EMBEDDING_PATH, delimiter=',')

    # efficientnet
    CFG.LOSS_MODULE = CFG.LOSS_MODULES[1]
    CFG.MODEL_NAME = CFG.MODEL_NAMES[3]
    CFG.MARGIN = CFG.MARGINS[0]
    CFG.MODEL_PATH = f'{CFG.MODEL_NAME}_{CFG.LOSS_MODULE}_face_epoch_8_bs_8_margin_{CFG.MARGIN}.pt'

    TEST_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_test_embed.csv'
    print(f'Loading {TEST_EMBEDDING_PATH} ...')
    test_embeddings_4 = np.loadtxt(TEST_EMBEDDING_PATH, delimiter=',')
    VALID_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_valid_embed.csv'
    print(f'Loading {VALID_EMBEDDING_PATH} ...')
    valid_embeddings_4 = np.loadtxt(VALID_EMBEDDING_PATH, delimiter=',')

    # nfnet
    CFG.LOSS_MODULE = CFG.LOSS_MODULES[0]
    CFG.MODEL_NAME = CFG.MODEL_NAMES[4]
    CFG.MARGIN = CFG.MARGINS[4]
    CFG.MODEL_PATH = f'{CFG.MODEL_NAME}_{CFG.LOSS_MODULE}_face_epoch_8_bs_8_margin_{CFG.MARGIN}.pt'

    TEST_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_test_embed.csv'
    print(f'Loading {TEST_EMBEDDING_PATH} ...')
    test_embeddings_5 = np.loadtxt(TEST_EMBEDDING_PATH, delimiter=',')
    VALID_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_valid_embed.csv'
    print(f'Loading {VALID_EMBEDDING_PATH} ...')
    valid_embeddings_5 = np.loadtxt(VALID_EMBEDDING_PATH, delimiter=',')

    # get voting of valid embeddings
    valid_embeddings_dict = {'emb_0':valid_embeddings_1, 
                             'emb_1':valid_embeddings_2,
                             'emb_2':valid_embeddings_3,
                             'emb_3':valid_embeddings_4,
                             'emb_4':valid_embeddings_5}
    distances, indices = get_voting_nns(valid_embeddings_dict)

    # search best thresholds
    search_voting_threshold(valid_df, distances, indices)

    # get voting of test embeddings
    test_embeddings_dict = {'emb_0':test_embeddings_1, 
                            'emb_1':test_embeddings_2,
                            'emb_2':test_embeddings_3,
                            'emb_3':test_embeddings_4,
                            'emb_4':test_embeddings_5}
    distances, indices = get_voting_nns(test_embeddings_dict)

    # use obtained thresholds to get test results
    image_predictions = get_voting_predictions(test_df, distances, indices, threshold = CFG.BEST_THRESHOLD, min2 = False)
    # use obtained thresholds to get test results after min2
    image_predictions_min2 = get_voting_predictions(test_df, distances, indices, threshold = CFG.BEST_THRESHOLD_MIN2, min2 = True)

    ##################################### text ####################################
    valid_embeddings_dict = {}
    test_embeddings_dict = {}
    for i in range(len(CFG.BERT_MODEL_NAMES)):
        CFG.BERT_MODEL_NAME = CFG.BERT_MODEL_NAMES[i]
        CFG.MARGIN = CFG.BEST_BERT_MARGINS[0][i]
        model = ShopeeBertModel(
            model_name = CFG.BERT_MODEL_NAME,
            margin = CFG.MARGIN
        )
        CFG.MODEL_PATH_BERT = f"{CFG.BERT_MODEL_NAME.rsplit('/', 1)[-1]}_epoch8-bs16x1_margin_{CFG.MARGIN}.pt"
        SAVE_MODEL_PATH = CFG.TEXT_MODEL_PATH_PREFIX + CFG.MODEL_PATH_BERT
        print(CFG.MODEL_PATH_BERT)
        model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=CFG.DEVICE))
        valid_embeddings_dict[f'emb_{i}'] = get_bert_embeddings(valid_df, 'title', model)
        test_embeddings_dict[f'emb_{i}'] = get_bert_embeddings(test_df, 'title', model)

    distances, indices = get_voting_nns(valid_embeddings_dict)

    # search best thresholds
    search_voting_threshold(valid_df, distances, indices)
    distances, indices = get_voting_nns(test_embeddings_dict)

    # use obtained thresholds to get test results
    text_predictions = get_voting_predictions(test_df, distances, indices, threshold = CFG.BEST_THRESHOLD, min2 = False)
    # use obtained thresholds to get test results after min2
    text_predictions_min2 = get_voting_predictions(test_df, distances, indices, threshold = CFG.BEST_THRESHOLD_MIN2, min2 = True)
    
    #################################### union predictions ###################################
    result_list = [[0 for i in range(3)] for j in range(2)]

    test_df['image_predictions'] = image_predictions
    test_df['text_predictions'] = text_predictions
    test_df['pred_matches'] = test_df.apply(combine_predictions, axis = 1)
    test_df['f1'] = f1_score(test_df['matches'], test_df['pred_matches'])
    test_df['recall'] = recall_score(test_df['matches'], test_df['pred_matches'])
    test_df['precision'] = precision_score(test_df['matches'], test_df['pred_matches'])
    test_f1 = test_df.f1.mean()
    test_recall = test_df.recall.mean()
    test_precision = test_df.precision.mean()
    result_list[0][0] = test_f1
    result_list[0][1] = test_recall
    result_list[0][2] = test_precision
    print(f'Test f1 score = {test_f1}, recall = {test_recall}, precision = {test_precision}')

    # Min2
    test_df['image_predictions'] = image_predictions_min2
    test_df['text_predictions'] = text_predictions_min2
    test_df['pred_matches'] = test_df.apply(combine_predictions, axis = 1)
    test_df['f1'] = f1_score(test_df['matches'], test_df['pred_matches'])
    test_df['recall'] = recall_score(test_df['matches'], test_df['pred_matches'])
    test_df['precision'] = precision_score(test_df['matches'], test_df['pred_matches'])
    test_f1 = test_df.f1.mean()
    test_recall = test_df.recall.mean()
    test_precision = test_df.precision.mean()
    result_list[1][0] = test_f1
    result_list[1][1] = test_recall
    result_list[1][2] = test_precision
    print(f'Test f1 score after min2 = {test_f1}, recall = {test_recall}, precision = {test_precision}')

    result_df = pd.DataFrame(result_list,columns=['f1','recall','precision'])

    return result_df
