import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from config import CFG

def get_train_transforms():
    return albumentations.Compose(
        [   
            albumentations.Resize(CFG.IMG_SIZE,CFG.IMG_SIZE,always_apply=True),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=120, p=0.8),
            albumentations.RandomBrightness(limit=(0.09, 0.6), p=0.5),
            albumentations.Normalize(mean = CFG.MEAN, std = CFG.STD),
            ToTensorV2(p=1.0),
        ]
    )

def get_valid_transforms():

    return albumentations.Compose(
        [
            albumentations.Resize(CFG.IMG_SIZE,CFG.IMG_SIZE,always_apply=True),
            albumentations.Normalize(mean = CFG.MEAN, std = CFG.STD),
            ToTensorV2(p=1.0)
        ]
    )

def get_test_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(CFG.IMG_SIZE,CFG.IMG_SIZE,always_apply=True),
            albumentations.Normalize(mean=CFG.MEAN, std=CFG.STD),
            ToTensorV2(p=1.0)
        ]
    )
