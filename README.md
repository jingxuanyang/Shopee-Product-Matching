# Shopee product matching

## Introduction

This is the project of pattern recognition, with the problem chosen from a [Kaggle competition](https://www.kaggle.com/c/shopee-product-matching/overview).

## Structure

The directory of this project is as follows.

```s
|- bin/
    |-
|- input/
    |- 
|- test/
    |-
|- README.md
|- requirements.txt
```

## Quick Start

First, please check the following basic requirements. Note that you should have more GPU memory than the listed.

+ System: Ubuntu 18.04.4 LTS
+ Python: 3.8.3
+ GPU: Nvidia GeForce RTX 3080

Second, please install required packages included in `requirements.txt` via `pip`.

```s
pip install -r requirements.txt
```

Third, you can get the best image model trained and obtain the test results via following command. Also, you can double-click to open the file `bin/image_model_training.ipynb` and then run this file.

```s
jupyter nbconvert --execute --to notebook --inplace --allow-errors --ExecutePreprocessor.timeout=-1 image_model_training.ipynb
```

Finally, if you have any question in running the codes in this project, please do not hesitate to email me: `yangjx20@mails.tsinghua.edu.cn`.
