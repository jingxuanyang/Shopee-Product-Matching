import gc
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
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

def get_voting_neighbors(df, distances, indices, threshold = 0.2, min2 = False):
    predictions = []
    for k in range(distances.shape[0]):
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

    return df

def get_voting_nns(embeddings_dict):
    embs_num = len(embeddings_dict)
    similarities_sum = 0.
    for i in range(embs_num):
        try:
            emb = normalize(embeddings_dict[f'emb_{i}'])
        except KeyError:
            raise KeyError('Please use keys emb_0, emb_1, etc in embeddings dict.')
        similarities = emb.dot(emb.T)
        similarities_sum += similarities
    similarities_sum = similarities_sum / embs_num
    similarities = np.sort(similarities_sum)[:,:-51:-1]
    distances = 1 - similarities
    indices = np.argsort(similarities_sum)[:,:-51:-1]

    return distances, indices

def get_voting_result(df, distances, indices):
    predictions = []
    for k in range(distances.shape[0]):
        idx = np.where(distances[k,] < CFG.BEST_THRESHOLD)[0]
        ids = indices[k,idx]
        if len(ids) <= 1 and distances[k,1] < CFG.BEST_THRESHOLD_MIN2:
            ids = np.append(ids,indices[k,1])
        posting_ids = ' '.join(df['posting_id'].iloc[ids].values)
        predictions.append(posting_ids)
        
    df['pred_matches'] = predictions
    df['f1'] = f1_score(df['matches'], df['pred_matches'])
    df['recall'] = recall_score(df['matches'], df['pred_matches'])
    df['precision'] = precision_score(df['matches'], df['pred_matches'])

    f1 = df.f1.mean()
    recall = df.recall.mean()
    precision = df.precision.mean()
    print(f'f1 score after voting = {f1}, recall = {recall}, precision = {precision}')

    return df

def get_union_neighbors(df, embeddings, threshold = 0.2, min2 = False):

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
        predictions.append(df['posting_id'].iloc[ids].values)
    
    del nbrs, distances, indices
    gc.collect()
    return predictions
