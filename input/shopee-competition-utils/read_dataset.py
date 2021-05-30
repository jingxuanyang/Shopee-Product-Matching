import pandas as pd
from config import CFG
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold

def read_dataset():
    df = pd.read_csv(CFG.TRAIN_CSV)
    df['matches'] = df.label_group.map(df.groupby('label_group').posting_id.agg('unique').to_dict())
    df['matches'] = df['matches'].apply(lambda x: ' '.join(x))

    gkf = GroupKFold(n_splits=CFG.N_SPLITS)
    df['fold'] = -1
    for i, (train_idx, valid_idx) in enumerate(gkf.split(X=df, groups=df['label_group'])):
        df.loc[valid_idx, 'fold'] = i

    labelencoder= LabelEncoder()
    df['label_group'] = labelencoder.fit_transform(df['label_group'])

    train_df = df[df['fold']!=CFG.TEST_FOLD].reset_index(drop=True)
    train_df = train_df[train_df['fold']!=CFG.VALID_FOLD].reset_index(drop=True)
    valid_df = df[df['fold']==CFG.VALID_FOLD].reset_index(drop=True)

    if CFG.USE_TEST_CSV:
        test_df = pd.read_csv(CFG.TEST_CSV)
        test_df['matches'] = test_df.label_group.map(test_df.groupby('label_group').posting_id.agg('unique').to_dict())
        test_df['matches'] = test_df['matches'].apply(lambda x: ' '.join(x))
        test_df['label_group'] = labelencoder.fit_transform(test_df['label_group'])
    else:
        test_df = df[df['fold']==CFG.TEST_FOLD].reset_index(drop=True)

    train_df['label_group'] = labelencoder.fit_transform(train_df['label_group'])

    return train_df, valid_df, test_df
