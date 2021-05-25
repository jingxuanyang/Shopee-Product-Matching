import numpy as np
from config import CFG
from get_embeddings import get_image_embeddings
from get_neighbors import get_image_neighbors
from read_dataset import read_dataset

def search_best_threshold(model=0):
    _, valid_df, _ = read_dataset()
    search_space = np.arange(10, 50, 1)
    if CFG.USE_EMBEDDING:
        VALID_EMBEDDING_PATH = CFG.EMB_PATH_PREFIX + CFG.MODEL_PATH[:-3] + '_valid_embed.csv'
        valid_embeddings = np.loadtxt(VALID_EMBEDDING_PATH, delimiter=',')
    else:
        try:
            valid_embeddings = get_image_embeddings(valid_df, model)
        except:
            raise('Please load neural network model and including it in input.')

    print("Searching best threshold...")
    best_f1_valid = 0.
    best_threshold = 0.
    for i in search_space:
        threshold = i / 100
        valid_df = get_image_neighbors(valid_df, valid_embeddings, threshold=threshold)
        valid_f1 = valid_df.f1.mean()
        valid_recall = valid_df.recall.mean()
        valid_precision = valid_df.precision.mean()
        print(f"threshold = {threshold} -> f1 score = {valid_f1}, recall = {valid_recall}, precision = {valid_precision}")
        if (valid_f1 > best_f1_valid):
            best_f1_valid = valid_f1
            best_threshold = threshold

    print("Best threshold =", best_threshold)
    print("Best f1 score =", best_f1_valid)
    CFG.BEST_THRESHOLD = best_threshold

    # phase 2 search
    print("________________________________")
    print("Searching best min2 threshold...")
    search_space = np.arange(CFG.BEST_THRESHOLD * 100, CFG.BEST_THRESHOLD * 100 + 20, 0.5)

    best_f1_valid = 0.
    best_threshold = 0.

    for i in search_space:
        threshold = i / 100
        valid_df = get_image_neighbors(valid_df, valid_embeddings, threshold=threshold,min2=True)

        valid_f1 = valid_df.f1.mean()
        valid_recall = valid_df.recall.mean()
        valid_precision = valid_df.precision.mean()

        print(f"min2 threshold = {threshold} -> f1 score = {valid_f1}, recall = {valid_recall}, precision = {valid_precision}")

        if (valid_f1 > best_f1_valid):
            best_f1_valid = valid_f1
            best_threshold = threshold

    print("Best min2 threshold =", best_threshold)
    print("Best f1 score after min2 =", best_f1_valid)
    CFG.BEST_THRESHOLD_MIN2 = best_threshold

def search_inb_threshold(valid_df,valid_embeddings):
    search_space = np.arange(5, 45, 1)
    print("Searching best threshold...")
    best_f1_valid = 0.
    best_threshold = 0.
    for i in search_space:
        threshold = i / 100
        valid_df = get_image_neighbors(valid_df, valid_embeddings, threshold=threshold)
        valid_f1 = valid_df.f1.mean()
        valid_recall = valid_df.recall.mean()
        valid_precision = valid_df.precision.mean()
        print(f"threshold = {threshold} -> f1 score = {valid_f1}, recall = {valid_recall}, precision = {valid_precision}")
        if (valid_f1 > best_f1_valid):
            best_f1_valid = valid_f1
            best_threshold = threshold

    print("Best threshold =", best_threshold)
    print("Best f1 score =", best_f1_valid)
    CFG.BEST_THRESHOLD = best_threshold

    # phase 2 search
    print("________________________________")
    print("Searching best min2 threshold...")
    search_space = np.arange(CFG.BEST_THRESHOLD * 100, CFG.BEST_THRESHOLD * 100 + 20, 0.5)

    best_f1_valid = 0.
    best_threshold = 0.

    for i in search_space:
        threshold = i / 100
        valid_df = get_image_neighbors(valid_df, valid_embeddings, threshold=threshold,min2=True)

        valid_f1 = valid_df.f1.mean()
        valid_recall = valid_df.recall.mean()
        valid_precision = valid_df.precision.mean()

        print(f"min2 threshold = {threshold} -> f1 score = {valid_f1}, recall = {valid_recall}, precision = {valid_precision}")

        if valid_f1 > best_f1_valid:
            best_f1_valid = valid_f1
            best_threshold = threshold

    print("Best min2 threshold =", best_threshold)
    print("Best f1 score after min2 =", best_f1_valid)
    CFG.BEST_THRESHOLD_MIN2 = best_threshold
