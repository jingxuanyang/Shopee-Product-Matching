import torch
import numpy as np
from config import CFG
from read_dataset import read_dataset
from shopee_image_model import ShopeeModel
from custom_activation import replace_activations, Mish
from get_embeddings import get_image_embeddings
from get_neighbors import get_image_neighbors
from search_threshold import search_best_threshold, search_inb_threshold
from blend_neighborhood import blend_neighborhood

def run_test():
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
    """Test all models
    """
    _, _, test_df = read_dataset()
    for i in range(len(CFG.LOSS_MODULES)):
        CFG.LOSS_MODULE = CFG.LOSS_MODULES[i]
        if 'arc' in CFG.LOSS_MODULE:
            CFG.USE_ARCFACE = True
        else:
            CFG.USE_ARCFACE = False
        for j in range(len(CFG.MODEL_NAMES)):
            CFG.MODEL_NAME = CFG.MODEL_NAMES[j]
            for k in range(len(CFG.MARGINS)):
                CFG.MARGIN = CFG.MARGINS[k]
                if CFG.USE_EMBEDDING:
                    search_best_threshold()
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
                    model.load_state_dict(torch.load(CFG.MODEL_PATH))
                    model = model.to(CFG.DEVICE)

                    search_best_threshold(model)
                    test_embeddings = get_image_embeddings(test_df, model)

                test_df = get_image_neighbors(test_df, test_embeddings, threshold=CFG.BEST_THRESHOLD)
                test_f1 = test_df.f1.mean()
                test_recall = test_df.recall.mean()
                test_precision = test_df.precision.mean()
                print(f'Test f1 score = {test_f1}, recall = {test_recall}, precision = {test_precision}')

                test_df = get_image_neighbors(test_df, test_embeddings, threshold=CFG.BEST_THRESHOLD_MIN2, min2 = True)
                test_f1 = test_df.f1.mean()
                test_recall = test_df.recall.mean()
                test_precision = test_df.precision.mean()
                print(f'Test f1 score after min2 = {test_f1}, recall = {test_recall}, precision = {test_precision}')
