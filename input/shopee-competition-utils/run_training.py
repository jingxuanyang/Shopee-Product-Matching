import torch
from transformers import AdamW
from read_dataset import read_dataset
from dataset import ShopeeDataset, TitleDataset
from augmentations import get_train_transforms
from config import CFG
from shopee_image_model import ShopeeModel
from shopee_text_model import ShopeeBertModel
from custom_activation import replace_activations, Mish
from custom_optimizer import Ranger, SAM
from custom_scheduler import ShopeeScheduler, GradualWarmupSchedulerV2
from engine import train_fn, train_text_fn
from get_embeddings import get_valid_embeddings, get_bert_embeddings
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

def run_bert_training():
    seed_everything(CFG.SEED_BERT)
    train_df, valid_df, _ = read_dataset()

    train_dataset = TitleDataset(train_df, 'title', 'label_group')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = CFG.BATCH_SIZE,
        num_workers = CFG.NUM_WORKERS,
        pin_memory = True,
        shuffle = True,
        drop_last = True
    )

    model = ShopeeBertModel(
        model_name = CFG.BERT_MODEL_NAME,
        margin = CFG.MARGIN
    )
    model.to(CFG.DEVICE)

    # Create Optimizer
    optimizer_grouped_parameters = [
        {'params': model.backbone.parameters(), 'lr': CFG.SCHEDULER_PARAMS_BERT['lr_start']},
        {'params': model.classifier.parameters(), 'lr': CFG.SCHEDULER_PARAMS_BERT['lr_start'] * 2},
        {'params': model.bn.parameters(), 'lr': CFG.SCHEDULER_PARAMS_BERT['lr_start'] * 2},
        {'params': model.final.parameters(), 'lr': CFG.SCHEDULER_PARAMS_BERT['lr_start'] * 2},
    ]

    if CFG.USE_SAM:
        optimizer = AdamW
        optimizer = SAM(optimizer_grouped_parameters, optimizer)
    else:
        optimizer = AdamW(optimizer_grouped_parameters)

    # Create Scheduler
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.COSINE_EPO-2, eta_min=CFG.ETA_MIN, last_epoch=-1)
    scheduler = GradualWarmupSchedulerV2(optimizer, multiplier=CFG.MULTIPLIER, total_epoch=CFG.WARMUP_EPO,after_scheduler=scheduler_cosine)

    max_f1_valid = 0.
    for epoch in range(CFG.EPOCHS):
        model, _ = train_text_fn(model, train_dataloader, optimizer, scheduler,
                                 CFG.USE_SAM, CFG.ACCUM_ITER, epoch, CFG.DEVICE, CFG.USE_AMP)

        valid_embeddings = get_bert_embeddings(valid_df, 'title', model)
        valid_df, _ = get_valid_neighbors(valid_df, valid_embeddings, threshold=CFG.BERT_KNN_THRESHOLD)

        valid_f1 = valid_df.f1.mean()
        valid_recall = valid_df.recall.mean()
        valid_precision = valid_df.precision.mean()
        print(f'Valid f1 score = {valid_f1}, recall = {valid_recall}, precision = {valid_precision}')

        if (epoch >= CFG.MIN_SAVE_EPOCH) and (valid_f1 > max_f1_valid):
            SAVE_MODEL_PATH = CFG.TEXT_MODEL_PATH_PREFIX + CFG.MODEL_PATH_BERT
            print(f"Valid f1 score improved. Saving model weights to {SAVE_MODEL_PATH}")
            max_f1_valid = valid_f1
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
