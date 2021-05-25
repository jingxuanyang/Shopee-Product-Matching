import gc
import torch
import numpy as np
from augmentations import get_test_transforms, get_valid_transforms
from dataset import ShopeeImageDataset
from config import CFG
from tqdm.notebook import tqdm

def get_image_embeddings(df, model):

    image_dataset = ShopeeImageDataset(df,transform=get_test_transforms())
    image_loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=CFG.BATCH_SIZE,
        pin_memory=True,
        num_workers = CFG.NUM_WORKERS,
        drop_last=False
    )

    embeds = []
    with torch.no_grad():
        for img,label in tqdm(image_loader): 
            img = img.to(CFG.DEVICE)
            label = label.to(CFG.DEVICE)
            feat,_ = model(img,label)
            image_embeddings = feat.detach().cpu().numpy()
            embeds.append(image_embeddings)
    
    del model
    image_embeddings = np.concatenate(embeds)
    print(f'Our image embeddings shape is {image_embeddings.shape}')
    del embeds
    gc.collect()
    return image_embeddings

def get_valid_embeddings(df, model):

    model.eval()

    image_dataset = ShopeeImageDataset(df,transform=get_valid_transforms())
    image_loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=CFG.BATCH_SIZE,
        pin_memory=True,
        num_workers = CFG.NUM_WORKERS,
        drop_last=False
    )

    embeds = []
    with torch.no_grad():
        for img,label in tqdm(image_loader): 
            img = img.to(CFG.DEVICE)
            label = label.to(CFG.DEVICE)
            feat,_ = model(img,label)
            image_embeddings = feat.detach().cpu().numpy()
            embeds.append(image_embeddings)
    
    del model
    image_embeddings = np.concatenate(embeds)
    print(f'Our image embeddings shape is {image_embeddings.shape}')
    del embeds
    gc.collect()
    return image_embeddings
