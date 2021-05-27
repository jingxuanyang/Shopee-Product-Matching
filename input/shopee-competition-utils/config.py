class CFG: 
    
    DATA_DIR = '../input/shopee-product-matching/train_images'
    TRAIN_CSV = '../input/shopee-product-matching/train.csv'

    # data augmentation
    IMG_SIZE = 512
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    SEED = 2021

    # data split
    N_SPLITS = 5
    TEST_FOLD = 0
    VALID_FOLD = 1

    EPOCHS = 8
    BATCH_SIZE = 8

    NUM_WORKERS = 4
    DEVICE = 'cuda:0'

    CLASSES = 6609 
    SCALE = 30

    MARGIN = 0.5
    MARGINS = [0.5,0.6,0.7,0.8,0.9]

    BEST_THRESHOLD = 0.19
    BEST_THRESHOLD_MIN2 = 0.225

    MODEL_NAME = 'resnet50'
    MODEL_NAMES = ['resnet50','resnext50_32x4d','densenet121','efficientnet_b3','eca_nfnet_l0']

    LOSS_MODULE = 'arc'
    LOSS_MODULES = ['arc','curricular']

    USE_ARCFACE = True
    USE_EMBEDDING = True

    MODEL_PATH_PREFIX = '../input/image-model-trained/'
    EMB_PATH_PREFIX = '../input/image-embeddings/'
    RESULTS_SAVE_PATH = '../input/shopee-competition-results/'
    TEXT_MODEL_PATH_PREFIX = '../input/text-model-trained/'
    
    MODEL_PATH = f'{MODEL_NAME}_{LOSS_MODULE}_face_epoch_8_bs_8_margin_{MARGIN}.pt'

    FC_DIM = 512

    SCHEDULER_PARAMS = {
            "lr_start": 1e-5,
            "lr_max": 1e-5 * 32,
            "lr_min": 1e-6,
            "lr_ramp_ep": 5,
            "lr_sus_ep": 0,
            "lr_decay": 0.8,
        }

    ################################### BERT ####################################
    BERT_MODEL_NAMES = ['bert-base-multilingual-uncased',
                        'cahya/bert-base-indonesian-1.5G',
                        'cahya/distilbert-base-indonesian',
                        'sentence-transformers/paraphrase-xlm-r-multilingual-v1',
                        'sentence-transformers/paraphrase-distilroberta-base-v1']

    BERT_MODEL_NAME = 'sentence-transformers/paraphrase-xlm-r-multilingual-v1'

    MAX_LENGTH = 128

    BERT_MARGINS = [0.5,0.6,0.7,0.8]
    FC_DIM_BERT = 768
    SEED_BERT = 412
    CLASSES_BERT = 6609
    
    # Training
    ACCUM_ITER = 1  # 1 if use_sam = True
    MIN_SAVE_EPOCH = EPOCHS // 3
    USE_SAM = True  # SAM (Sharpness-Aware Minimization for Efficiently Improving Generalization)
    USE_AMP = True  # Automatic Mixed Precision
    
    # NearestNeighbors
    BERT_KNN_THRESHOLD = 0.4  # Cosine distance threshold
    
    # GradualWarmupSchedulerV2（lr_start -> lr_max -> lr_min）
    SCHEDULER_PARAMS_BERT = {
        "lr_start": 7.5e-6,
        "lr_max": 1e-4,
        "lr_min": 2.74e-5, # 1.5e-5,
    }

    MULTIPLIER = SCHEDULER_PARAMS_BERT['lr_max'] / SCHEDULER_PARAMS_BERT['lr_start']
    ETA_MIN = SCHEDULER_PARAMS_BERT['lr_min']  # last minimum learning rate
    FREEZE_EPO = 0
    WARMUP_EPO = 2
    COSINE_EPO = EPOCHS - FREEZE_EPO - WARMUP_EPO
    
    # save_model_path
    MODEL_PATH_BERT = f"{BERT_MODEL_NAME.rsplit('/', 1)[-1]}_epoch{EPOCHS}-bs{BATCH_SIZE}x{ACCUM_ITER}_margin_{MARGIN}.pt"
