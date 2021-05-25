import gc
import numpy as np
from sklearn.neighbors import NearestNeighbors
from config import CFG
from criterion import precision_score, recall_score, f1_score

def get_image_neighbors(df, embeddings, threshold = 0.2, min2 = False):

    nbrs = NearestNeighbors(n_neighbors = 50, metric = 'cosine')
    nbrs.fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    predictions = []
    for k in range(embeddings.shape[0]):
        if min2:
            idx = np.where(distances[k,] < CFG.BEST_THRESHOLD)[0]
            ids = indices[k,idx]
            if len(ids) <= 1 and distances[k,1] < threshold:
                ids = np.append(ids,indices[k,1])
        else:
            idx = np.where(distances[k,] < threshold)[0]
            ids = indices[k,idx]
        posting_ids = ' '.join(df['posting_id'].iloc[ids].values)
        predictions.append(posting_ids)
        
    df['pred_matches'] = predictions
    df['f1'] = f1_score(df['matches'], df['pred_matches'])
    df['recall'] = recall_score(df['matches'], df['pred_matches'])
    df['precision'] = precision_score(df['matches'], df['pred_matches'])
    
    del nbrs, distances, indices
    gc.collect()
    return df

def get_valid_neighbors(df, embeddings, KNN = 50, threshold = 0.36):

    nbrs = NearestNeighbors(n_neighbors = KNN, metric = 'cosine')
    nbrs.fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    predictions = []
    for k in range(embeddings.shape[0]):
        idx = np.where(distances[k,] < threshold)[0]
        ids = indices[k,idx]
        posting_ids = ' '.join(df['posting_id'].iloc[ids].values)
        predictions.append(posting_ids)
        
    df['pred_matches'] = predictions
    df['f1'] = f1_score(df['matches'], df['pred_matches'])
    df['recall'] = recall_score(df['matches'], df['pred_matches'])
    df['precision'] = precision_score(df['matches'], df['pred_matches'])
    
    del nbrs, distances, indices
    gc.collect()
    return df, predictions
