<!-- Author: Jingxuan Yang -->
<!-- E-mail: yangjx20@mails.tsinghua.edu.cn -->

# Shopee product matching

![GitHub Version](https://img.shields.io/github/release/jingxuanyang/Shopee-Product-Matching.svg)
![Repo Size](https://img.shields.io/github/repo-size/jingxuanyang/Shopee-Product-Matching.svg)

## Introduction

This is the project of pattern recognition, with the problem chosen from a [Kaggle competition](https://www.kaggle.com/c/shopee-product-matching/overview).

## Structure

The directory of this project is as follows.

```s
|- bin/
|- input/
  |- shopee-competition-utils/
  |- shopee-product-matching/
  |- text-model-trained/
|- notebook/
|- notebook-image-text/
|- notebook-text/
|- test/
|- README.md
|- requirements.txt
```

## File Descriptions

+ bin/
  + This folder contains the executable code files
+ input/
  + This folder includes all data and utilities of this project
+ input/shopee-competition-utils/
  + This folder contains utilities of this project
+ input/shopee-product-matching/
  + This folder contains all data files of this project
+ input/text-model-trained/
  + This folder stores trained text model (the best single model)
+ notebook/
  + This folder contains jupyter notebooks for training and inference of image models
+ notebook-image-text/
  + This folder contains jupyter notebooks for ensembles of image and text model 
+ notebook-text/
  + This folder contains jupyter notebooks for training and inference of text models
+ test/
  + This folder includes test jupyter notebooks
+ README.md
  + This file serves as user manual of this project
+ requirements.txt
  + This file contains python packages used in this project

## Quick Start

First, please check the following basic requirements. Note that you should have GPU memory >= 10 GiB.

+ System: Ubuntu 18.04.4 LTS
+ Python: 3.8.3
+ GPU: Nvidia GeForce RTX 3080

Then, please install required packages included in `requirements.txt` via `pip`.

```s
~$ pip install -r requirements.txt
```

Second, download Shopee dataset into specific directory via `wget` and unzip it via `unzip`.

```s
~$ cd input/shopee-product-matching/
~/input/shopee-product-matching$ wget -O shopee-dataset https://cloud.tsinghua.edu.cn/f/5c7ba8c55e04478d86d9/?dl=1
~/input/shopee-product-matching$ unzip shopee-dataset
```

Third, download pretrained model into specific directory via `wget` and unzip it via `unzip`.

```s
~/input/shopee-product-matching$ cd ..
~/input$ cd text-model-trained/
~/input/text-model-trained$ wget -O pretrained-model https://cloud.tsinghua.edu.cn/f/a6df401f7fd34248a7f9/?dl=1
~/input/text-model-trained$ unzip pretrained-model
```

Then, you can get inference results predicted by the best performing pretrained model stored in `input/text-model-trained/paraphrase-xlm-r-multilingual-v1_epoch8-bs16x1_margin_0.8.pt` via following command.

```s
~/input/text-model-trained$ cd ..
~/input$ cd ..
~$ cd bin/
~/bin$ python best_model_inference.py
```

Expected output is listed as follows.

```python
2021-05-30 15:01:17.692841: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
Building Model Backbone for sentence-transformers/paraphrase-xlm-r-multilingual-v1 model, margin = 0.8
paraphrase-xlm-r-multilingual-v1_epoch8-bs16x1_margin_0.8.pt
get_bert_embeddings: 100%|████████████████████| 216/216 [00:04<00:00, 53.66it/s]
Searching best threshold...
threshold = 0.01 -> f1 score = 0.5027464921403513, recall = 0.36920819465293997, precision = 0.9980143086581982
threshold = 0.02 -> f1 score = 0.51908882853155, recall = 0.38723567961541383, precision = 0.9974181145666033
threshold = 0.03 -> f1 score = 0.5349359255261192, recall = 0.4057179921329236, precision = 0.996348441552121
threshold = 0.04 -> f1 score = 0.5483479866781269, recall = 0.4214936699090649, precision = 0.9959288460602522
threshold = 0.05 -> f1 score = 0.5615151720270062, recall = 0.4369220353693416, precision = 0.9951892525151396
threshold = 0.06 -> f1 score = 0.572522496832261, recall = 0.4499498851171465, precision = 0.9941684222236128
threshold = 0.07 -> f1 score = 0.5828269001282261, recall = 0.4630646540637196, precision = 0.9928750460616429
threshold = 0.08 -> f1 score = 0.5923589345616298, recall = 0.47527675084010945, precision = 0.9918085017625099
threshold = 0.09 -> f1 score = 0.6028817489215734, recall = 0.48869024336912714, precision = 0.9896568146896665
threshold = 0.1 -> f1 score = 0.6140175003791694, recall = 0.5028970367148061, precision = 0.9877276140416749
threshold = 0.11 -> f1 score = 0.62386027143292, recall = 0.5154268014490383, precision = 0.9860473200814861
threshold = 0.12 -> f1 score = 0.6329263729093964, recall = 0.5272049660782009, precision = 0.984572014915861
threshold = 0.13 -> f1 score = 0.6415742308480433, recall = 0.5385016389206072, precision = 0.9825331933522911
threshold = 0.14 -> f1 score = 0.6509542129922906, recall = 0.5505961898060125, precision = 0.9806906117681413
threshold = 0.15 -> f1 score = 0.6613230553681155, recall = 0.5637775207575203, precision = 0.9796416230384691
threshold = 0.16 -> f1 score = 0.6707598680559608, recall = 0.576318907369572, precision = 0.9780584035292751
threshold = 0.17 -> f1 score = 0.6797794458855402, recall = 0.588864470205251, precision = 0.9756061806981647
threshold = 0.18 -> f1 score = 0.6893698890550152, recall = 0.601842795280555, precision = 0.9738874730661853
threshold = 0.19 -> f1 score = 0.6977106466800822, recall = 0.6135103219186495, precision = 0.9715956493661273
threshold = 0.2 -> f1 score = 0.705557075500399, recall = 0.6244402770499733, precision = 0.9684070249116241
threshold = 0.21 -> f1 score = 0.7148727538141246, recall = 0.6371927778462941, precision = 0.9658399046484898
threshold = 0.22 -> f1 score = 0.7230899823376746, recall = 0.6487996833863309, precision = 0.9637884795619662
threshold = 0.23 -> f1 score = 0.7312534197718673, recall = 0.660648264078936, precision = 0.9613948045324894
threshold = 0.24 -> f1 score = 0.7396437624371135, recall = 0.6724451072826095, precision = 0.9592029407622928
threshold = 0.25 -> f1 score = 0.747392900496289, recall = 0.6841417369370686, precision = 0.9562429090468643
threshold = 0.26 -> f1 score = 0.7538771362762666, recall = 0.6945884160419811, precision = 0.9531772548970213
threshold = 0.27 -> f1 score = 0.759814282983688, recall = 0.7047343504781323, precision = 0.9493948631602904
threshold = 0.28 -> f1 score = 0.7665991383537749, recall = 0.7153999383729984, precision = 0.9469576309760154
threshold = 0.29 -> f1 score = 0.7736130938560478, recall = 0.7272729195702812, precision = 0.9432776941237178
threshold = 0.3 -> f1 score = 0.779379946678424, recall = 0.7372108343822307, precision = 0.9397291852517998
threshold = 0.31 -> f1 score = 0.785689762548553, recall = 0.7495067019249689, precision = 0.9346678389263784
threshold = 0.32 -> f1 score = 0.7901121078804931, recall = 0.7591831656237933, precision = 0.9294421088474973
threshold = 0.33 -> f1 score = 0.7937219879211048, recall = 0.7685087876110084, precision = 0.9231231060422324
threshold = 0.34 -> f1 score = 0.7981994636517731, recall = 0.7784750110025295, precision = 0.9181930097375393
threshold = 0.35 -> f1 score = 0.8021934727794962, recall = 0.788138692028146, precision = 0.913231398437369
threshold = 0.36 -> f1 score = 0.8043672987939667, recall = 0.7966856405948681, precision = 0.9060587264079084
threshold = 0.37 -> f1 score = 0.8072324527609893, recall = 0.8049980232592455, precision = 0.9001233290473561
threshold = 0.38 -> f1 score = 0.8082144423775688, recall = 0.8127577760142956, precision = 0.8925303891940227
threshold = 0.39 -> f1 score = 0.809614292997088, recall = 0.8219766435514532, precision = 0.8837434600443329
threshold = 0.4 -> f1 score = 0.8099172083192242, recall = 0.8304567849311008, precision = 0.8735989203634735
threshold = 0.41 -> f1 score = 0.8108409376275685, recall = 0.8388584092063919, precision = 0.8659611695732178
threshold = 0.42 -> f1 score = 0.8095973922481051, recall = 0.8462106247388016, precision = 0.8552502716038131
Best threshold = 0.41
Best f1 score = 0.8108409376275685
________________________________
Searching best min2 threshold...
min2 threshold = 0.41 -> f1 score = 0.8108409376275685, recall = 0.8388584092063919, precision = 0.8659611695732178
min2 threshold = 0.415 -> f1 score = 0.8116496526062353, recall = 0.8403184763694813, precision = 0.8650851292753641
min2 threshold = 0.42 -> f1 score = 0.8119074227720214, recall = 0.8410038967877096, precision = 0.8639170755448925
min2 threshold = 0.425 -> f1 score = 0.8123004735811915, recall = 0.8417610459022831, precision = 0.8630410352470388
min2 threshold = 0.43 -> f1 score = 0.8130044523622618, recall = 0.8427455483322522, precision = 0.8625300117399575
min2 threshold = 0.435 -> f1 score = 0.813370627936497, recall = 0.8435850869510287, precision = 0.8616539714421039
min2 threshold = 0.44 -> f1 score = 0.8137780562020066, recall = 0.8444051580076306, precision = 0.860923937860559
min2 threshold = 0.445 -> f1 score = 0.8141418339201425, recall = 0.8450152575007785, precision = 0.8604129143534777
min2 threshold = 0.45 -> f1 score = 0.814340224942011, recall = 0.8454739619345158, precision = 0.8597558841300874
min2 threshold = 0.455 -> f1 score = 0.8147394991728866, recall = 0.846241621896734, precision = 0.8590258505485426
min2 threshold = 0.46 -> f1 score = 0.8152679169592982, recall = 0.8471638976547522, precision = 0.8581498102506889
min2 threshold = 0.465 -> f1 score = 0.8158717663147901, recall = 0.8481178082013042, precision = 0.8575657833854531
min2 threshold = 0.47 -> f1 score = 0.8160661996454475, recall = 0.8486886249350645, precision = 0.8566897430875993
min2 threshold = 0.475 -> f1 score = 0.8162040916361648, recall = 0.8492118156685049, precision = 0.8555946927152823
min2 threshold = 0.48 -> f1 score = 0.8164556529915369, recall = 0.8498128766506434, precision = 0.8548646591337374
min2 threshold = 0.485 -> f1 score = 0.8168963205149902, recall = 0.8506159135903425, precision = 0.8543536356266561
min2 threshold = 0.49 -> f1 score = 0.8172026120389777, recall = 0.8511919448496853, precision = 0.8536966054032658
min2 threshold = 0.495 -> f1 score = 0.8170725965344552, recall = 0.8516591663418739, precision = 0.8520175281657126
Best min2 threshold = 0.49
Best f1 score after min2 = 0.8172026120389777
get_bert_embeddings: 100%|████████████████████| 216/216 [00:03<00:00, 61.97it/s]
Test f1 score = 0.8211137129836418, recall = 0.8483214531160035, precision = 0.8724217517245711
Test f1 score after min2 = 0.8285356189862213, recall = 0.8627090001536798, precision = 0.8590660372303366
Searching best threshold...
threshold = 0.01 -> f1 score = 0.7487150548288662, recall = 0.6982848267193497, precision = 0.94961581998194
threshold = 0.02 -> f1 score = 0.7729656472412717, recall = 0.7452999015407432, precision = 0.9285892501433977
threshold = 0.03 -> f1 score = 0.7883627503081633, recall = 0.7729596755071844, precision = 0.9176470919190673
threshold = 0.04 -> f1 score = 0.799624631051321, recall = 0.7941936289620009, precision = 0.9081190304415624
threshold = 0.05 -> f1 score = 0.807271975602742, recall = 0.8103105651668139, precision = 0.8999584102541189
threshold = 0.06 -> f1 score = 0.8119752656268617, recall = 0.8213284726087994, precision = 0.8922592984801269
threshold = 0.07 -> f1 score = 0.815824492694392, recall = 0.8300783159368174, precision = 0.8874437182905293
threshold = 0.08 -> f1 score = 0.8174777546207815, recall = 0.8363261240829811, precision = 0.882693432678336
threshold = 0.09 -> f1 score = 0.8191989132279784, recall = 0.8413581814356539, precision = 0.8793886151244932
threshold = 0.1 -> f1 score = 0.8200570524655117, recall = 0.8456916825158566, precision = 0.8755977688010829
threshold = 0.11 -> f1 score = 0.820444773296711, recall = 0.8500903856971065, precision = 0.8716004677210466
threshold = 0.12 -> f1 score = 0.8207133839305376, recall = 0.8527055624654046, precision = 0.8690345416491833
threshold = 0.13 -> f1 score = 0.8207064158101671, recall = 0.8556725213105225, precision = 0.8660346294793263
Best threshold = 0.12
Best f1 score = 0.8207133839305376
________________________________
Searching best min2 threshold...
min2 threshold = 0.12 -> f1 score = 0.8207133839305376, recall = 0.8527055624654046, precision = 0.8690345416491833
min2 threshold = 0.125 -> f1 score = 0.8207949622863164, recall = 0.8529671578321247, precision = 0.8686695248584108
min2 threshold = 0.13 -> f1 score = 0.8208703604636274, recall = 0.8530701112859325, precision = 0.8685965215002562
min2 threshold = 0.135 -> f1 score = 0.820943363821782, recall = 0.8532161180022413, precision = 0.8685235181421018
min2 threshold = 0.14 -> f1 score = 0.8211591305239287, recall = 0.8535100969098862, precision = 0.8682315047094838
min2 threshold = 0.145 -> f1 score = 0.8212838102762912, recall = 0.8536840882468212, precision = 0.867939491276866
min2 threshold = 0.15 -> f1 score = 0.8212804034529105, recall = 0.8537147708176397, precision = 0.8677204812024025
Best min2 threshold = 0.145
Best f1 score after min2 = 0.8212838102762912
CFG.BEST_THRESHOLD after INB is 0.12
CFG.BEST_THRESHOLD_MIN2 after INB is 0.145
Test f1 score after INB = 0.834544833312457, recall = 0.8656430033625991, precision = 0.8770051228664779
```

Among the long output, the following three lines are of the most importance, which give inference results on test dataset.

```python
Test f1 score = 0.8211137129836418, recall = 0.8483214531160035, precision = 0.8724217517245711
Test f1 score after min2 = 0.8285356189862213, recall = 0.8627090001536798, precision = 0.8590660372303366
Test f1 score after INB = 0.834544833312457, recall = 0.8656430033625991, precision = 0.8770051228664779
```

Fourth, you can get the best image model trained via following command. Note that once you run this command, the stored model `input/text-model-trained/paraphrase-xlm-r-multilingual-v1_epoch8-bs16x1_margin_0.8.pt` will be replaced by newly trained one.

```s
~/bin$ python best_model_train.py
```

Fifth, you can get inference results predicted by this newly trained model via following command.

```s
~/bin$ python best_model_inference.py
```

What's more, you can also use this trained model to make inference on test file `input/shopee-product-matching/test.csv` via following command. Note that `test.csv` should be replaced with a file like `train.csv`. Specifically, `test.csv` should possess `label_group` property and include more than 50 samples, otherwise the following python script can not compute f1 score and other criteria, and thus you will get an error.

```s
~/bin$ python best_model_inference_using_test_csv.py
```

Finally, if you have any question in running the codes in this project, please do not hesitate to email me: `yangjx20@mails.tsinghua.edu.cn`.
