## canlpy

### Requirements:

Pytorch  
Python3 >=3.8  

### Setup

```Run pip install -e .``` in the home directory to setup the project

## Pre-trained Models

### ERNIE

Download pre-trained knowledge embedding from [Google Drive](https://drive.google.com/open?id=14VNvGMtYWxuqT-PWDa8sD0e7hO486i8Y)/[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/229e8cccedc2419f987e/) and extract it.

```shell
tar -xvzf kg_embed.tar.gz
```

Store the content of kg_embed in canlpy/knowledge/ernie/

Download pre-trained ERNIE from [Google Drive](https://drive.google.com/uc?export=download&id=1Hdp_iqsF3xjFcWSRvklC5ppvvd2C0qim) and extract it.

```shell
tar -xvzf ernie_base.tar.gz
```
Store the content of ernie_base in pretrained_models/ernie/

## Tokens

Add your tokens in a tokens.py file located in the canlpy/helpers folder

### Tagme
Token name is TAGME_TOKEN

## Tests

Tests can be run directly using python, eg: ```python3 -v ernie_test.py```
