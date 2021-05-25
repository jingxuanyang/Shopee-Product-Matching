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

df = pd.read_csv('../input/shopee-product-matching/train.csv')
df_cu = cudf.DataFrame(df)
