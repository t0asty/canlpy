## canlpy

A library reimplementing Knowledge Enhanced Language Models under a common Fusion abstraction. So far ERNIE, CokeBERT and KnowBert are the three provided models.

### Requirements:

PyTorch  
Python3 >=3.8

### Setup

Run ```pip install -e .``` in the home directory to setup the project
canlpy uses [spacy](https://spacy.io) for various text processing purposes, which requires a spacy language package to be installed using
```python -m spacy download en_core_web_sm```

## Pre-trained Models

### ERNIE

Download pre-trained knowledge embedding from [Google Drive](https://drive.google.com/open?id=14VNvGMtYWxuqT-PWDa8sD0e7hO486i8Y)

```shell
tar -xvzf kg_embed.tar.gz
```

Store the content of kg_embed in canlpy/knowledge/ernie/

Download pre-trained ERNIE from [Google Drive](https://drive.google.com/file/d/1cvUbXYGhRRCTWlewOuniQ7K7YIGy46PI) and extract it.

```shell
tar -xvzf ernie_base.tar.gz
```
Store the content of ernie_base in pretrained_models/ernie/

### Examples

Code to use the three models can be found in [examples/](examples/)

### Benchmarks

Benchmarks to compare the models can be found in [experiments/](experiments/)

### Tokens

To use ERNIE and CokeBERT, register your TAGME_TOKEN token in a tokens.py file located in the canlpy/helpers folder

### Acknowledgements

A portion of this code has been adapted from [ERNIE: Enhanced Language Representation with Informative Entities](https://github.com/thunlp/ERNIE), [CokeBERT: Contextual Knowledge Selection and Embedding towards Enhanced Pre-Trained Language Models](https://github.com/thunlp/CokeBERT), [KnowBert: Knowledge Enhanced Contextual Word Representations](https://github.com/allenai/kb)


