import os
import cv2
import torch
from config import CFG


class ShopeeDataset(torch.utils.data.Dataset):
    """for training
    """
    def __init__(self,df, transform = None):
        self.df = df 
        self.root_dir = CFG.DATA_DIR
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):

        row = self.df.iloc[idx]

        img_path = os.path.join(self.root_dir,row.image)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = row.label_group

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return {
            'image' : image,
            'label' : torch.tensor(label).long()
        }


class ShopeeImageDataset(torch.utils.data.Dataset):
    """for validating and test
    """
    def __init__(self,df, transform = None):
        self.df = df 
        self.root_dir = CFG.DATA_DIR
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):

        row = self.df.iloc[idx]

        img_path = os.path.join(self.root_dir,row.image)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = row.label_group

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']     
                
        return image,torch.tensor(1)

class TitleDataset(torch.utils.data.Dataset):
    def __init__(self, df, text_column, label_column):
        texts = df[text_column]
        self.labels = df[label_column].values
        
        self.titles = []
        for title in texts:
            title = title.encode('utf-8').decode("unicode_escape")
            title = title.encode('ascii', 'ignore').decode("unicode_escape")
            title = title.lower()
            self.titles.append(title)

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        text = self.titles[idx]
        label = torch.tensor(self.labels[idx])
        return text, label
        