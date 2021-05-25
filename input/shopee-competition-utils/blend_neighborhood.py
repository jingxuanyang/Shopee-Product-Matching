from sklearn.neighbors import NearestNeighbors
import numpy as np
from config import CFG
from criterion import precision_score, recall_score, f1_score

def blend_neighborhood(df, embeddings):
    nbrs = NearestNeighbors(n_neighbors = 50, metric = 'cosine')
    nbrs.fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    similarities = 1 - distances

    predictions = []
    new_emb = embeddings.copy()
    for k in range(embeddings.shape[0]):
        idx = np.where(distances[k,] < CFG.BEST_THRESHOLD)[0]
        ids = indices[k,idx]
        if len(ids) <= 1 and distances[k,1] < CFG.BEST_THRESHOLD_MIN2:
            ids = np.append(ids,indices[k,1])
        cur_emb = embeddings[ids]
        weights = np.expand_dims(similarities[k,0:len(ids)], 1)
        new_emb[k] = (cur_emb * weights).sum(axis=0)
    #     posting_ids = ' '.join(df['posting_id'].iloc[ids].values)
    #     predictions.append(posting_ids)
        
    # df['pred_matches'] = predictions
    # df['f1'] = f1_score(df['matches'], df['pred_matches'])
    # df['recall'] = recall_score(df['matches'], df['pred_matches'])
    # df['precision'] = precision_score(df['matches'], df['pred_matches'])

    # f1 = df.f1.mean()
    # recall = df.recall.mean()
    # precision = df.precision.mean()
    # print(f'f1 score = {f1}, recall = {recall}, precision = {precision}')

    return new_emb
