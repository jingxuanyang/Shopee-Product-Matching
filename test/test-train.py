import sys
sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')

# Preliminaries
from tqdm import tqdm # progress bar
import math
import random
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Visuals and CV2
import cv2

# albumentations for augs
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

# PyTorch Image Models
import timm

# torch
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

# configure
DIM = (512,512)

NUM_WORKERS = 4
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 30
SEED = 2020
LR = 3e-4 # Learning Rate

device = torch.device('cuda')


################################################# MODEL ####################################################################

model_name = 'efficientnet_b3' #efficientnet_b0-b7

################################################ Metric Loss and its params #######################################################
loss_module = 'arcface' #'cosface' #'adacos'
s = 30.0
m = 0.5 
ls_eps = 0.0
easy_margin = False


####################################### Scheduler and its params ############################################################
SCHEDULER = 'CosineAnnealingWarmRestarts' #'CosineAnnealingLR'
factor=0.2 # ReduceLROnPlateau
patience=4 # ReduceLROnPlateau
eps=1e-6 # ReduceLROnPlateau
T_max=10 # CosineAnnealingLR
T_0=4 # CosineAnnealingWarmRestarts
min_lr=1e-6

############################################## Model Params ###############################################################
model_params = {
    'n_classes':11014,
    'model_name':'efficientnet_b3',
    'use_fc':False,
    'fc_dim':512,
    'dropout':0.0,
    'loss_module':loss_module,
    's':30.0,
    'margin':0.50,
    'ls_eps':0.0,
    'theta_zero':0.785,
    'pretrained':True
}

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(SEED)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# methods on how to reduce learning rate
def fetch_scheduler(optimizer):
        if SCHEDULER =='ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True, eps=eps)
        elif SCHEDULER =='CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=min_lr, last_epoch=-1)
        elif SCHEDULER =='CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=min_lr, last_epoch=-1)
        return scheduler

# use cross entropy to compute loss
def fetch_loss():
    loss = nn.CrossEntropyLoss()
    return loss

# transform images for training
def get_train_transforms():
    return albumentations.Compose(
        [   
            albumentations.Resize(DIM[0],DIM[1],always_apply=True),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=120, p=0.8),
            albumentations.RandomBrightness(limit=(0.09, 0.6), p=0.5),
            #albumentations.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5),
            #albumentations.ShiftScaleRotate(
              #  shift_limit=0.25, scale_limit=0.1, rotate_limit=0
            #),
            albumentations.Normalize(),
            ToTensorV2(p=1.0),
        ]
    )

# transform images for validating
def get_valid_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(DIM[0],DIM[1],always_apply=True),
            albumentations.Normalize(),
            ToTensorV2(p=1.0)
        ]
    )

class ShopeeDataset(Dataset):
    '''Obtain image file with labels'''
    def __init__(self, csv, transforms=None):

        self.csv = csv.reset_index()
        self.augmentations = transforms

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        
        text = row.title
        
        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']       
                
        return image,torch.tensor(row.label_group)

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
                 pretrained=True):
        """
        :param n_classes:
        :param model_name: name of model from pretrainedmodels
            e.g. resnet50, resnext101_32x4d, pnasnet5large
        :param pooling: One of ('SPoC', 'MAC', 'RMAC', 'GeM', 'Rpool', 'Flatten', 'CompactBilinearPooling')
        :param loss_module: One of ('arcface', 'cosface', 'softmax')
        """
        super(ShopeeNet, self).__init__()
        print('Building Model Backbone for {} model'.format(model_name))

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        final_in_features = self.backbone.classifier.in_features
        
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        
        self.pooling = nn.AdaptiveAvgPool2d(1)
            
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
        return logits

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

def train_fn(dataloader,model,criterion,optimizer,device,scheduler,epoch):
    model.train()
    loss_score = AverageMeter()
    
    tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
    for bi,d in tk0:
        
        batch_size = d[0].shape[0]

        images = d[0]
        targets = d[1]

        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        output = model(images,targets)
        
        loss = criterion(output,targets)
        
        loss.backward()
        optimizer.step()
        
        loss_score.update(loss.detach().item(), batch_size)

        # add parameters to the progress bar
        tk0.set_postfix(Train_Loss=loss_score.avg,Epoch=epoch,LR=optimizer.param_groups[0]['lr'])
        
    if scheduler is not None:
        scheduler.step()
        
    return loss_score

def eval_fn(data_loader,model,criterion,device):
    
    loss_score = AverageMeter()
    
    model.eval()
    tk0 = tqdm(enumerate(data_loader), total=len(data_loader))
    
    with torch.no_grad():
        
        for bi,d in tk0:
            batch_size = d[0].size()[0]

            image = d[0]
            targets = d[1]

            image = image.to(device)
            targets = targets.to(device)

            output = model(image,targets)

            loss = criterion(output,targets)
            
            loss_score.update(loss.detach().item(), batch_size)

            tk0.set_postfix(Eval_Loss=loss_score.avg)
            
    return loss_score

# load train data with 5 folds
data = pd.read_csv('../input/shopee-folds/folds.csv')

# add image file path to data
data['filepath'] = data['image'].apply(lambda x: os.path.join('../input/shopee-product-matching/', 'train_images', x))

# show data
data.head()

# transform labels to [0, 11014 - 1]
encoder = LabelEncoder()
data['label_group'] = encoder.fit_transform(data['label_group'])

def run():
        
    train = data[data['fold']!=0].reset_index(drop=True)
    valid = data[data['fold']==0].reset_index(drop=True)

    # Defining DataSet
    train_dataset = ShopeeDataset(
        csv=train,
        transforms=get_train_transforms(),
    )
        
    valid_dataset = ShopeeDataset(
        csv=valid,
        transforms=get_valid_transforms(),
    )
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        pin_memory=True,
        drop_last=True,
        num_workers=NUM_WORKERS
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=VALID_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    
    # Defining Device
    device = torch.device("cuda:1")
    
    # Defining Model for specific fold
    model = ShopeeNet(**model_params)
    
    model = torch.nn.DataParallel(model, device_ids=[1,2,3])
    model.to(device)
    
    # Defining criterion: cross entropy loss
    criterion = fetch_loss()
    criterion.to(device)
    
    # Defining Optimizer with weight decay to params other than bias and layer norms
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]  
    
    optimizer = torch.optim.Adam(optimizer_parameters, lr=LR)
    
    # Defining Learning Rate SCheduler
    scheduler = fetch_scheduler(optimizer)
    
    # THE ENGINE LOOP
    best_loss = 10000
    for epoch in range(EPOCHS):
        train_loss = train_fn(train_loader, model, criterion, optimizer, device, scheduler=scheduler, epoch=epoch)
        
        valid_loss = eval_fn(valid_loader, model, criterion, device)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.module.state_dict(),f'model_{model_name}_IMG_SIZE_{DIM[0]}_{loss_module}.bin')
            print('best model found for epoch {}'.format(epoch))

run()
