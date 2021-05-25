import sys
sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')

# Preliminaries
from tqdm import tqdm
import math
import random
import os
import pandas as pd
import numpy as np

# Visuals and CV2
import cv2

# albumentations for augs
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

#torch
import torch
import timm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader

import gc
import matplotlib.pyplot as plt

import cudf
import cuml
import cupy
from cuml.feature_extraction.text import TfidfVectorizer
from cuml import PCA
from cuml.neighbors import NearestNeighbors

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import PCA
# from sklearn.neighbors import NearestNeighbors

DIM = (512,512)

NUM_WORKERS = 4
BATCH_SIZE = 16
SEED = 2021

device = torch.device('cuda')

CLASSES = 11014

# ADJUSTING FOR CV OR SUBMIT
CHECK_SUB = False
GET_CV = True

test = pd.read_csv('../input/shopee-product-matching/test.csv')

if len(test) > 3: 
    GET_CV = False
else: 
    print('this submission notebook will compute CV score, but commit notebook will not')

# MODEL
model_name = 'efficientnet_b3' #efficientnet_b0-b7

# MODEL PATH
IMG_MODEL_PATH = 'model_efficientnet_b3_IMG_SIZE_512_arcface.bin'
# IMG_MODEL_PATH = '../input/pytorch-metric-learning-pipeline-only-images/model_efficientnet_b3_IMG_SIZE_512_arcface.bin'

# Metric Loss and its params
loss_module = 'arcface' #'cosface' #'adacos'
s = 30.0
m = 0.5 
ls_eps = 0.0
easy_margin = False

def read_dataset():
    if GET_CV:
        df = pd.read_csv('../input/shopee-product-matching/train.csv')
        tmp = df.groupby(['label_group'])['posting_id'].unique().to_dict()
        df['matches'] = df['label_group'].map(tmp)
        df['matches'] = df['matches'].apply(lambda x: ' '.join(x))
        if CHECK_SUB:
            df = pd.concat([df, df], axis = 0)
            df.reset_index(drop = True, inplace = True)
        df_cu = cudf.DataFrame(df)
        # df_cu = df
        image_paths = '../input/shopee-product-matching/train_images/' + df['image']
    else:
        df = pd.read_csv('../input/shopee-product-matching/test.csv')
        df_cu = cudf.DataFrame(df)
        # df_cu = df
        image_paths = '../input/shopee-product-matching/test_images/' + df['image']
        
    return df, df_cu, image_paths

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(SEED)

def precision_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    len_y_pred = y_pred.apply(lambda x: len(x)).values
    precision = intersection / len_y_pred
    return precision

def recall_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    len_y_true = y_true.apply(lambda x: len(x)).values
    recall = intersection / len_y_true
    return recall

def f1_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    len_y_pred = y_pred.apply(lambda x: len(x)).values
    len_y_true = y_true.apply(lambda x: len(x)).values
    f1 = 2 * intersection / (len_y_pred + len_y_true)
    return f1

def combine_predictions(row):
    x = np.concatenate([row['image_predictions'], row['text_predictions']])
    return ' '.join( np.unique(x) )

def get_neighbors(df, embeddings, KNN = 50, image = True):
    '''
    https://www.kaggle.com/ragnar123/unsupervised-baseline-arcface?scriptVersionId=57121538
    '''

    model = NearestNeighbors(n_neighbors = KNN)
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)
    
    # Iterate through different thresholds to maximize cv, run this in interactive mode, then replace else clause with a solid threshold
    if GET_CV:
        if image:
            thresholds = list(np.arange(2,4,0.1))
        else:
            thresholds = list(np.arange(0.1, 1, 0.1))
        scores_f1 = []
        scores_recall = []
        scores_precision = []
        for threshold in thresholds:
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

            score_f1 = df['f1'].mean()
            score_recall = df['recall'].mean()
            score_precision = df['precision'].mean()

            print(f'Our f1 score for threshold {threshold} is {score_f1}, recall score is {score_recall}, mAP score is {score_precision}')

            scores_f1.append(score_f1)
            scores_recall.append(score_recall)
            scores_precision.append(score_precision)

        # create a dataframe to store threshold and scores
        thresholds_scores = pd.DataFrame({'thresholds': thresholds, 'scores_f1': scores_f1, 'scores_recall': scores_recall, 'scores_precision': scores_precision})
        
        # obtain best f1 score
        max_score = thresholds_scores[thresholds_scores['scores_f1'] == thresholds_scores['scores_f1'].max()]

        # obtain best threshold and scores
        best_threshold = max_score['thresholds'].values[0]
        best_score_f1 = max_score['scores_f1'].values[0]
        best_score_recall = max_score['scores_recall'].values[0]
        best_score_precision = max_score['scores_precision'].values[0]

        print(f'Our best f1 score is {best_score_f1} and has a threshold {best_threshold}, corresponding recall score is {best_score_recall}, mAP score is {best_score_precision}')
        
        # Use threshold
        predictions = []
        for k in range(embeddings.shape[0]):
            # Because we are predicting the test set that have 70K images and different label groups, confidence should be smaller
            if image:
                idx = np.where(distances[k,] < 2.7)[0]
            else:
                idx = np.where(distances[k,] < 0.60)[0]
            ids = indices[k,idx]
            posting_ids = df['posting_id'].iloc[ids].values
            predictions.append(posting_ids)
    
    # Because we are predicting the test set that have 70K images and different label groups, confidence should be smaller
    else:
        predictions = []
        for k in tqdm(range(embeddings.shape[0])):
            if image:
                idx = np.where(distances[k,] < 2.7)[0]
            else:
                idx = np.where(distances[k,] < 0.60)[0]
            ids = indices[k,idx]
            posting_ids = df['posting_id'].iloc[ids].values
            predictions.append(posting_ids)
        
    del model, distances, indices
    gc.collect()
    return df, predictions

def get_test_transforms():

    return albumentations.Compose(
        [
            albumentations.Resize(DIM[0],DIM[1],always_apply=True),
            albumentations.Normalize(),
            ToTensorV2(p=1.0)
        ]
    )

class ShopeeDataset(Dataset):
    def __init__(self, image_paths, transforms=None):

        self.image_paths = image_paths
        self.augmentations = transforms

    def __len__(self):
        return self.image_paths.shape[0]

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']       
        
        
        return image,torch.tensor(1)

class ShopeeNet(nn.Module):

    def __init__(self,
                 n_classes,
                 model_name='efficientnet_b0',
                 use_fc=False,
                 fc_dim=512,
                 dropout=0.0,
                 loss_module='softmax',
                 s=30.0,
                 margin=0.50,
                 ls_eps=0.0,
                 theta_zero=0.785,
                 pretrained=False):
        """
        :param n_classes:
        :param model_name: name of model from pretrainedmodels
            e.g. resnet50, resnext101_32x4d, pnasnet5large
        :param pooling: One of ('SPoC', 'MAC', 'RMAC', 'GeM', 'Rpool', 'Flatten', 'CompactBilinearPooling')
        :param loss_module: One of ('arcface', 'cosface', 'softmax')
        """
        super(ShopeeNet, self).__init__()
        print('Model building for {} backbone'.format(model_name))

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        final_in_features = self.backbone.classifier.in_features
        
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        
        self.pooling =  nn.AdaptiveAvgPool2d(1)
            
        self.use_fc = use_fc
        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            final_in_features = fc_dim

        self.loss_module = loss_module
        if loss_module == 'arcface':
            self.final = ArcMarginProduct(final_in_features, n_classes,
                                          s=s, m=margin, easy_margin=False, ls_eps=ls_eps)
        elif loss_module == 'cosface':
            self.final = AddMarginProduct(final_in_features, n_classes, s=s, m=margin)
        elif loss_module == 'adacos':
            self.final = AdaCos(final_in_features, n_classes, m=margin, theta_zero=theta_zero)
        else:
            self.final = nn.Linear(final_in_features, n_classes)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, label):
        feature = self.extract_feat(x)
        if self.loss_module in ('arcface', 'cosface', 'adacos'):
            logits = self.final(feature, label)
        else:
            logits = self.final(feature)
        return feature,logits

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)

        return x

class AdaCos(nn.Module):
    def __init__(self, in_features, out_features, m=0.50, ls_eps=0, theta_zero=math.pi/4):
        super(AdaCos, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.theta_zero = theta_zero
        self.s = math.log(out_features - 1) / math.cos(theta_zero)
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.weight)
        # dot product
        logits = F.linear(x, W)
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            theta_med = torch.median(theta)
            self.s = torch.log(B_avg) / torch.cos(torch.min(self.theta_zero * torch.ones_like(theta_med), theta_med))
        output *= self.s

        return output

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

def get_image_embeddings(image_paths):
    embeds = []
    
    model = ShopeeNet(n_classes=CLASSES,model_name=model_name)
    model.eval()
    
    model.load_state_dict(torch.load(IMG_MODEL_PATH),strict=False)
    model = model.to(device)

    image_dataset = ShopeeDataset(image_paths=image_paths,transforms=get_test_transforms())
    image_loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        drop_last=False,
        num_workers=NUM_WORKERS
    )
    
    
    with torch.no_grad():
        for img,label in tqdm(image_loader): 
            img = img.cuda()
            label = label.cuda()
            feat, _ = model(img,label)
            image_embeddings = feat.detach().cpu().numpy()
            embeds.append(image_embeddings)
    
    
    del model
    image_embeddings = np.concatenate(embeds)
    print(f'Our image embeddings shape is {image_embeddings.shape}')
    del embeds
    gc.collect()
    return image_embeddings

def get_text_embeddings(df_cu, max_features = 15000, n_components = 5000):
    model = TfidfVectorizer(stop_words = 'english', binary = True, max_features = max_features)
    text_embeddings = model.fit_transform(df_cu['title']).toarray()
    pca = PCA(n_components = n_components)
    text_embeddings = pca.fit_transform(text_embeddings).get()
    # text_embeddings = pca.fit_transform(text_embeddings)
    print(f'Our title text embedding shape is {text_embeddings.shape}')
    del model, pca
    gc.collect()
    return text_embeddings

df,df_cu,image_paths = read_dataset()
print(df.head())

image_embeddings = get_image_embeddings(image_paths.values)
text_embeddings = get_text_embeddings(df_cu, max_features = 15000, n_components = 5000)

# Get neighbors for image_embeddings
df,image_predictions = get_neighbors(df, image_embeddings, KNN = 50, image = True)

print(df.head())

# Get neighbors for text_embeddings
df, text_predictions = get_neighbors(df, text_embeddings, KNN = 50, image = False)

print(df.head())

if GET_CV:
    df['image_predictions'] = image_predictions
    df['text_predictions'] = text_predictions
    df['pred_matches'] = df.apply(combine_predictions, axis = 1)
    df['f1'] = f1_score(df['matches'], df['pred_matches'])
    df['recall'] = recall_score(df['matches'], df['pred_matches'])
    df['precision'] = precision_score(df['matches'], df['pred_matches'])
    score_f1 = df['f1'].mean()
    score_recall = df['recall'].mean()
    score_precision = df['precision'].mean()
    print(f'Our final f1 cv score is {score_f1}, recall score is {score_recall}, mAP score is {score_precision}')
    df['matches'] = df['pred_matches']
    df[['posting_id', 'matches']].to_csv('submission.csv', index = False)
else:
    df['image_predictions'] = image_predictions
    df['text_predictions'] = text_predictions
    df['matches'] = df.apply(combine_predictions, axis = 1)
    df[['posting_id', 'matches']].to_csv('submission.csv', index = False)

