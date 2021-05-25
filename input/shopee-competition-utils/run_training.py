import torch
from read_dataset import read_dataset
from dataset import ShopeeDataset
from augmentations import get_train_transforms
from config import CFG
from shopee_image_model import ShopeeModel
from custom_activation import replace_activations, Mish
from custom_optimizer import Ranger
from custom_scheduler import ShopeeScheduler
from engine import train_fn
from get_embeddings import get_valid_embeddings
from get_neighbors import get_valid_neighbors
from seed_everything import seed_everything

def run_training():
    seed_everything(CFG.SEED)
    train_df, valid_df, _ = read_dataset()
    train_dataset = ShopeeDataset(train_df, transform = get_train_transforms())

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = CFG.BATCH_SIZE,
        pin_memory = True,
        num_workers = CFG.NUM_WORKERS,
        shuffle = True,
        drop_last = True
    )

    print('Training classes:', train_df['label_group'].nunique())
    print('Model saving path:', CFG.MODEL_PATH)

    if 'arc' in CFG.LOSS_MODULE:
        CFG.USE_ARCFACE = True
    else:
        CFG.USE_ARCFACE = False

    model = ShopeeModel(model_name = CFG.MODEL_NAME,
                        margin = CFG.MARGIN,
                        use_arcface = CFG.USE_ARCFACE)
    model = replace_activations(model, torch.nn.SiLU, Mish())
    model.to(CFG.DEVICE)

    optimizer = Ranger(model.parameters(), lr = CFG.SCHEDULER_PARAMS['lr_start'])
    scheduler = ShopeeScheduler(optimizer,**CFG.SCHEDULER_PARAMS)

    best_valid_f1 = 0.
    for i in range(CFG.EPOCHS):
        _ = train_fn(model, train_dataloader, optimizer, scheduler, i)

        valid_embeddings = get_valid_embeddings(valid_df, model)
        valid_df, _ = get_valid_neighbors(valid_df, valid_embeddings)

        valid_f1 = valid_df.f1.mean()
        valid_recall = valid_df.recall.mean()
        valid_precision = valid_df.precision.mean()
        print(f'Valid f1 score = {valid_f1}, recall = {valid_recall}, precision = {valid_precision}')

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            print('Valid f1 score improved, model saved')
            MODEL_PATH = CFG.MODEL_PATH_PREFIX + CFG.MODEL_PATH
            torch.save(model.state_dict(),MODEL_PATH)
