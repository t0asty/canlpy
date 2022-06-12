## canlpy

### Requirements:

Pytorch  
Python3 >=3.8  

### Setup

Run ```pip install -e .``` in the home directory to setup the project

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

### CokeBert

You can find the pretrained CokeBert model provided by the authors of the CokeBert paper, which is available from [here](https://drive.google.com/file/d/1Ce7Nq7vJ83l4lOV9SiiN2Kq831z_phsV/view?usp=sharing). 
Then unzip the file, and copy the content of the `DKPLM_BERTbase_2layer` folder into `canlpy/canlpy/pretrained_models/cokebert` 

From the same file, copy the `load_data_n` folder into `canlpy/canlpy/knowledge/cokebert/`

You can find the datasets for the knowledge graph representation used by the authors of the ERNIE paper, download them into this folder from [here](https://drive.google.com/open?id=1HlWw7Q6-dFSm9jNSCh4VaBf1PlGqt9im) and run 

```shell
tar -xvzf data.tar.gz
```
to extract it. 

Then copy `kg_embed` into `canlpy/canlpy/knowledge/cokebert/`

Add your TAGME token in a tokens.py file located in the canlpy/helpers folder  
Token name is TAGME_TOKEN

Once that is done, you can run `cokebert.ipynb` from the examples folder.

## Tokens

Add your tokens in a tokens.py file located in the canlpy/helpers folder

### Tagme
Token name is TAGME_TOKEN
