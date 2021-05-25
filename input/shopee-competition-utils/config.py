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

    MARGINS = [0.5,0.6,0.7,0.8,0.9]
    MARGIN = 0.5

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
