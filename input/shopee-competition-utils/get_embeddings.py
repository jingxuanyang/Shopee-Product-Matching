import gc
import torch
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from augmentations import get_test_transforms, get_valid_transforms
from dataset import ShopeeImageDataset
from config import CFG

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

def get_bert_embeddings(df, column, model, chunk=32):
    model.eval()
    
    bert_embeddings = torch.zeros((df.shape[0], 768)).to(CFG.DEVICE)
    for i in tqdm(list(range(0, df.shape[0], chunk)) + [df.shape[0]-chunk], desc="get_bert_embeddings", ncols=80):
        titles = []
        for title in df[column][i : i + chunk].values:
            try:
                title = title.encode('utf-8').decode("unicode_escape")
                title = title.encode('ascii', 'ignore').decode("unicode_escape")
            except:
                pass
            #title = text_punctuation(title)
            title = title.lower()
            titles.append(title)
            
        with torch.no_grad():
            if CFG.USE_AMP:
                with torch.cuda.amp.autocast():
                    model_output = model(titles)
            else:
                model_output = model(titles)
            
        bert_embeddings[i : i + chunk] = model_output
    bert_embeddings = bert_embeddings.detach().cpu().numpy()
    del model, titles, model_output
    gc.collect()
    torch.cuda.empty_cache()
    
    return bert_embeddings

def get_tfidf_embeddings(df, max_features = 15000):
    model = TfidfVectorizer(stop_words = 'english', binary = True, max_features = max_features)
    text_embeddings = model.fit_transform(df['title']).toarray()
    print(f'Our title text embedding shape is {text_embeddings.shape}')

    del model
    gc.collect()

    return text_embeddings
