import torch
import numpy as np
import pandas as pd
from config import CFG
from read_dataset import read_dataset
from shopee_image_model import ShopeeModel
from shopee_text_model import ShopeeBertModel
from custom_activation import replace_activations, Mish
from get_embeddings import get_image_embeddings, get_bert_embeddings
from get_neighbors import get_image_neighbors
from search_threshold import search_inb_threshold
from seed_everything import seed_everything
from blend_neighborhood import blend_neighborhood

def run_test():
    seed_everything(CFG.SEED)
    _, valid_df, test_df = read_dataset()

    if 'arc' in CFG.LOSS_MODULE:
        CFG.USE_ARCFACE = True
    else:
        CFG.USE_ARCFACE = False

    if CFG.USE_EMBEDDING:
        VALID_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_valid_embed.csv'
        valid_embeddings = np.loadtxt(VALID_EMBEDDING_PATH, delimiter=',')
        search_inb_threshold(valid_df,valid_embeddings)
        CFG.MODEL_PATH = f'{CFG.MODEL_NAME}_{CFG.LOSS_MODULE}_face_epoch_8_bs_8_margin_{CFG.MARGIN}.pt'
        TEST_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_test_embed.csv'
        test_embeddings = np.loadtxt(TEST_EMBEDDING_PATH, delimiter=',')
    else:
        model = ShopeeModel(model_name = CFG.MODEL_NAME,
                            margin = CFG.MARGIN,
                            use_arcface = CFG.USE_ARCFACE)
        model.eval()
        model = replace_activations(model, torch.nn.SiLU, Mish())
        CFG.MODEL_PATH = f'{CFG.MODEL_NAME}_{CFG.LOSS_MODULE}_face_epoch_8_bs_8_margin_{CFG.MARGIN}.pt'
        print(CFG.MODEL_PATH)
        MODEL_PATH = CFG.MODEL_PATH_PREFIX + CFG.MODEL_PATH
        model.load_state_dict(torch.load(MODEL_PATH))
        model = model.to(CFG.DEVICE)

        try:
            valid_embeddings = get_image_embeddings(valid_df, model)
        except:
            raise('Please load neural network model and including it in input.')

        search_inb_threshold(valid_df,valid_embeddings)
        test_embeddings = get_image_embeddings(test_df, model)

    test_df = get_image_neighbors(test_df, test_embeddings, threshold=CFG.BEST_THRESHOLD)
    test_f1 = test_df.f1.mean()
    test_recall = test_df.recall.mean()
    test_precision = test_df.precision.mean()
    print(f'Test f1 score = {test_f1}, recall = {test_recall}, precision = {test_precision}')

    # Min2
    test_df = get_image_neighbors(test_df, test_embeddings, threshold=CFG.BEST_THRESHOLD_MIN2, min2 = True)
    test_f1 = test_df.f1.mean()
    test_recall = test_df.recall.mean()
    test_precision = test_df.precision.mean()
    print(f'Test f1 score after min2 = {test_f1}, recall = {test_recall}, precision = {test_precision}')

    # INB
    new_test_emb = blend_neighborhood(test_df,test_embeddings)
    new_valid_emb = blend_neighborhood(valid_df,valid_embeddings)
    search_inb_threshold(valid_df,new_valid_emb)
    print(f'CFG.BEST_THRESHOLD after INB is {CFG.BEST_THRESHOLD}')
    print(f'CFG.BEST_THRESHOLD_MIN2 after INB is {CFG.BEST_THRESHOLD_MIN2}')

    test_df = get_image_neighbors(test_df, new_test_emb, threshold=CFG.BEST_THRESHOLD_MIN2, min2 = True)
    test_f1 = test_df.f1.mean()
    test_recall = test_df.recall.mean()
    test_precision = test_df.precision.mean()
    print(f'Test f1 score after INB = {test_f1}, recall = {test_recall}, precision = {test_precision}')

def run_test_all():
    """Test all models.
    :return test_result_dict: dictionary of all test results
    """
    seed_everything(CFG.SEED)
    _, valid_df, test_df = read_dataset()
    test_result_dict = {}
    for i in range(len(CFG.MODEL_NAMES)):
        CFG.MODEL_NAME = CFG.MODEL_NAMES[i]
        test_f1_list = [[0 for i in range(6)] for j in range(5)]
        test_recall_list = [[0 for i in range(6)] for j in range(5)]
        test_precision_list = [[0 for i in range(6)] for j in range(5)]
        for j in range(len(CFG.LOSS_MODULES)):
            CFG.LOSS_MODULE = CFG.LOSS_MODULES[j]
            if 'arc' in CFG.LOSS_MODULE:
                CFG.USE_ARCFACE = True
            else:
                CFG.USE_ARCFACE = False
            for k in range(len(CFG.MARGINS)):
                CFG.MARGIN = CFG.MARGINS[k]
                if CFG.USE_EMBEDDING:
                    CFG.MODEL_PATH = f'{CFG.MODEL_NAME}_{CFG.LOSS_MODULE}_face_epoch_8_bs_8_margin_{CFG.MARGIN}.pt'
                    print(CFG.MODEL_PATH)
                    VALID_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_valid_embed.csv'
                    valid_embeddings = np.loadtxt(VALID_EMBEDDING_PATH, delimiter=',')
                    search_inb_threshold(valid_df,valid_embeddings)
                    TEST_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_test_embed.csv'
                    test_embeddings = np.loadtxt(TEST_EMBEDDING_PATH, delimiter=',')
                else:
                    model = ShopeeModel(model_name = CFG.MODEL_NAME,
                                        margin = CFG.MARGIN,
                                        use_arcface = CFG.USE_ARCFACE)
                    model.eval()
                    model = replace_activations(model, torch.nn.SiLU, Mish())
                    CFG.MODEL_PATH = f'{CFG.MODEL_NAME}_{CFG.LOSS_MODULE}_face_epoch_8_bs_8_margin_{CFG.MARGIN}.pt'
                    print(CFG.MODEL_PATH)
                    MODEL_PATH = CFG.MODEL_PATH_PREFIX + CFG.MODEL_PATH
                    model.load_state_dict(torch.load(MODEL_PATH))
                    model = model.to(CFG.DEVICE)
                    valid_embeddings = get_image_embeddings(valid_df, model)
                    search_inb_threshold(valid_df,valid_embeddings)
                    test_embeddings = get_image_embeddings(test_df, model)

                test_df = get_image_neighbors(test_df, test_embeddings, threshold=CFG.BEST_THRESHOLD)
                test_f1 = test_df.f1.mean()
                test_recall = test_df.recall.mean()
                test_precision = test_df.precision.mean()
                test_f1_list[k][3*j] = test_f1
                test_recall_list[k][3*j] = test_recall
                test_precision_list[k][3*j] = test_precision
                print(f'Test f1 score = {test_f1}, recall = {test_recall}, precision = {test_precision}')

                # Min2
                test_df = get_image_neighbors(test_df, test_embeddings, threshold=CFG.BEST_THRESHOLD_MIN2, min2 = True)
                test_f1 = test_df.f1.mean()
                test_recall = test_df.recall.mean()
                test_precision = test_df.precision.mean()
                test_f1_list[k][3*j+1] = test_f1
                test_recall_list[k][3*j+1] = test_recall
                test_precision_list[k][3*j+1] = test_precision
                print(f'Test f1 score after min2 = {test_f1}, recall = {test_recall}, precision = {test_precision}')

                # INB
                new_test_emb = blend_neighborhood(test_df,test_embeddings)
                new_valid_emb = blend_neighborhood(valid_df,valid_embeddings)
                search_inb_threshold(valid_df,new_valid_emb)
                print(f'CFG.BEST_THRESHOLD after INB is {CFG.BEST_THRESHOLD}')
                print(f'CFG.BEST_THRESHOLD_MIN2 after INB is {CFG.BEST_THRESHOLD_MIN2}')

                test_df = get_image_neighbors(test_df, new_test_emb, threshold=CFG.BEST_THRESHOLD_MIN2, min2 = True)
                test_f1 = test_df.f1.mean()
                test_recall = test_df.recall.mean()
                test_precision = test_df.precision.mean()
                test_f1_list[k][3*j+2] = test_f1
                test_recall_list[k][3*j+2] = test_recall
                test_precision_list[k][3*j+2] = test_precision
                print(f'Test f1 score after INB = {test_f1}, recall = {test_recall}, precision = {test_precision}')

        test_f1_df = pd.DataFrame(test_f1_list,columns=['arcface','arcface_min2','arcface_inb','curface','curface_min2','curface_inb'])
        test_recall_df = pd.DataFrame(test_recall_list,columns=['arcface','arcface_min2','arcface_inb','curface','curface_min2','curface_inb'])
        test_precision_df = pd.DataFrame(test_precision_list,columns=['arcface','arcface_min2','arcface_inb','curface','curface_min2','curface_inb'])

        test_result_dict[f'{CFG.MODEL_NAME}_f1'] = test_f1_df
        test_result_dict[f'{CFG.MODEL_NAME}_recall'] = test_recall_df
        test_result_dict[f'{CFG.MODEL_NAME}_precision'] = test_precision_df

    return test_result_dict

# for BERT model
def run_bert_test():
    seed_everything(CFG.SEED_BERT)
    _, valid_df, test_df = read_dataset()

    model = ShopeeBertModel(
        model_name = CFG.BERT_MODEL_NAME,
        margin = CFG.MARGIN
    )
    SAVE_MODEL_PATH = CFG.TEXT_MODEL_PATH_PREFIX + CFG.MODEL_PATH_BERT
    print(CFG.MODEL_PATH_BERT)
    model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=CFG.DEVICE))
    valid_embeddings = get_bert_embeddings(valid_df, 'title', model)
    search_inb_threshold(valid_df,valid_embeddings)
    test_embeddings = get_bert_embeddings(test_df, 'title', model)

    test_df = get_image_neighbors(test_df, test_embeddings, threshold=CFG.BEST_THRESHOLD)
    test_f1 = test_df.f1.mean()
    test_recall = test_df.recall.mean()
    test_precision = test_df.precision.mean()
    print(f'Test f1 score = {test_f1}, recall = {test_recall}, precision = {test_precision}')

    # Min2
    test_df = get_image_neighbors(test_df, test_embeddings, threshold=CFG.BEST_THRESHOLD_MIN2, min2=True)
    test_f1 = test_df.f1.mean()
    test_recall = test_df.recall.mean()
    test_precision = test_df.precision.mean()
    print(f'Test f1 score after min2 = {test_f1}, recall = {test_recall}, precision = {test_precision}')

    # INB
    new_test_emb = blend_neighborhood(test_df,test_embeddings)
    new_valid_emb = blend_neighborhood(valid_df,valid_embeddings)
    search_inb_threshold(valid_df,new_valid_emb)
    print(f'CFG.BEST_THRESHOLD after INB is {CFG.BEST_THRESHOLD}')
    print(f'CFG.BEST_THRESHOLD_MIN2 after INB is {CFG.BEST_THRESHOLD_MIN2}')

    test_df = get_image_neighbors(test_df, new_test_emb, threshold=CFG.BEST_THRESHOLD_MIN2, min2 = True)
    test_f1 = test_df.f1.mean()
    test_recall = test_df.recall.mean()
    test_precision = test_df.precision.mean()
    print(f'Test f1 score after INB = {test_f1}, recall = {test_recall}, precision = {test_precision}')

def run_bert_test_all():
    """Test all models.
    :return test_result_dict: dictionary of all test results
    """
    seed_everything(CFG.SEED_BERT)
    _, valid_df, test_df = read_dataset()
    test_result_dict = {}
    for i in range(len(CFG.BERT_MODEL_NAMES)):
        CFG.MODEL_NAME = CFG.BERT_MODEL_NAMES[i]
        test_f1_list = [[0 for i in range(3)] for j in range(len(CFG.BERT_MARGINS))]
        test_recall_list = [[0 for i in range(3)] for j in range(len(CFG.BERT_MARGINS))]
        test_precision_list = [[0 for i in range(3)] for j in range(len(CFG.BERT_MARGINS))]
        for k in range(len(CFG.BERT_MARGINS)):
            CFG.MARGIN = CFG.BERT_MARGINS[k]
            if CFG.USE_EMBEDDING:
                CFG.MODEL_PATH_BERT = f"{CFG.BERT_MODEL_NAME.rsplit('/', 1)[-1]}_epoch8-bs16x1_margin_{CFG.MARGIN}.pt"
                print(CFG.MODEL_PATH_BERT)
                VALID_EMBEDDING_PATH = CFG.BERT_EMB_PATH_PREFIX + CFG.MODEL_PATH_BERT[:-3] + '_valid_embed.csv'
                valid_embeddings = np.loadtxt(VALID_EMBEDDING_PATH, delimiter=',')
                search_inb_threshold(valid_df,valid_embeddings)
                TEST_EMBEDDING_PATH = CFG.BERT_EMB_PATH_PREFIX + CFG.MODEL_PATH_BERT[:-3] + '_test_embed.csv'
                test_embeddings = np.loadtxt(TEST_EMBEDDING_PATH, delimiter=',')
            else:
                model = ShopeeBertModel(
                    model_name = CFG.BERT_MODEL_NAME,
                    margin = CFG.MARGIN
                )
                CFG.MODEL_PATH_BERT = f"{CFG.BERT_MODEL_NAME.rsplit('/', 1)[-1]}_epoch8-bs16x1_margin_{CFG.MARGIN}.pt"
                SAVE_MODEL_PATH = CFG.TEXT_MODEL_PATH_PREFIX + CFG.MODEL_PATH_BERT
                print(CFG.MODEL_PATH_BERT)
                model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=CFG.DEVICE))
                valid_embeddings = get_bert_embeddings(valid_df, 'title', model)
                search_inb_threshold(valid_df,valid_embeddings)
                test_embeddings = get_bert_embeddings(test_df, 'title', model)

            test_df = get_image_neighbors(test_df, test_embeddings, threshold=CFG.BEST_THRESHOLD)
            test_f1 = test_df.f1.mean()
            test_recall = test_df.recall.mean()
            test_precision = test_df.precision.mean()
            test_f1_list[k][0] = test_f1
            test_recall_list[k][0] = test_recall
            test_precision_list[k][0] = test_precision
            print(f'Test f1 score = {test_f1}, recall = {test_recall}, precision = {test_precision}')

            # Min2
            test_df = get_image_neighbors(test_df, test_embeddings, threshold=CFG.BEST_THRESHOLD_MIN2, min2 = True)
            test_f1 = test_df.f1.mean()
            test_recall = test_df.recall.mean()
            test_precision = test_df.precision.mean()
            test_f1_list[k][1] = test_f1
            test_recall_list[k][1] = test_recall
            test_precision_list[k][1] = test_precision
            print(f'Test f1 score after min2 = {test_f1}, recall = {test_recall}, precision = {test_precision}')

            # INB
            new_test_emb = blend_neighborhood(test_df,test_embeddings)
            new_valid_emb = blend_neighborhood(valid_df,valid_embeddings)
            search_inb_threshold(valid_df,new_valid_emb)
            print(f'CFG.BEST_THRESHOLD after INB is {CFG.BEST_THRESHOLD}')
            print(f'CFG.BEST_THRESHOLD_MIN2 after INB is {CFG.BEST_THRESHOLD_MIN2}')

            test_df = get_image_neighbors(test_df, new_test_emb, threshold=CFG.BEST_THRESHOLD_MIN2, min2 = True)
            test_f1 = test_df.f1.mean()
            test_recall = test_df.recall.mean()
            test_precision = test_df.precision.mean()
            test_f1_list[k][2] = test_f1
            test_recall_list[k][2] = test_recall
            test_precision_list[k][2] = test_precision
            print(f'Test f1 score after INB = {test_f1}, recall = {test_recall}, precision = {test_precision}')

        test_f1_df = pd.DataFrame(test_f1_list,columns=['arcface','arcface_min2','arcface_inb'])
        test_recall_df = pd.DataFrame(test_recall_list,columns=['arcface','arcface_min2','arcface_inb'])
        test_precision_df = pd.DataFrame(test_precision_list,columns=['arcface','arcface_min2','arcface_inb'])

        test_result_dict[f"{CFG.MODEL_NAME.rsplit('/', 1)[-1]}_f1"] = test_f1_df
        test_result_dict[f"{CFG.MODEL_NAME.rsplit('/', 1)[-1]}_recall"] = test_recall_df
        test_result_dict[f"{CFG.MODEL_NAME.rsplit('/', 1)[-1]}_precision"] = test_precision_df

    return test_result_dict
