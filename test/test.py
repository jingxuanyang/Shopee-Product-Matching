import pandas as pd
import numpy as np

data = {'match':['train_111', 'train_211', 'train_123'], 'pred_match':['train_111', 'train_121', 'train_123']}
df = pd.DataFrame(data)

y_true = df['match']
y_pred = df['pred_match']

y_true = y_true.apply(lambda x: set(x.split()))
y_pred = y_pred.apply(lambda x: set(x.split()))

intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
len_y_pred = y_pred.apply(lambda x: len(x)).values
len_y_true = y_true.apply(lambda x: len(x)).values
f1 = 2 * intersection / (len_y_pred + len_y_true)
f1_mean = f1.mean()
print(intersection)
print(len_y_pred)
print(len_y_true)
print(f1)
print(f1_mean)
