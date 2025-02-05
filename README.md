## canlpy

A library reimplementing Knowledge Enhanced Language Models under a common Fusion abstraction. So far ERNIE, CokeBERT and KnowBert are the three provided models.

### Requirements:

PyTorch  
Python3 >=3.8

### Setup

Run ```pip install -e .``` in the home directory to setup the project
canlpy uses [spacy](https://spacy.io) for various text processing purposes, which requires a spacy language package to be installed using
```python -m spacy download en_core_web_sm```

### Documentation

To generate the documentation, get the latest version of pdoc and run 

```pdoc ./canlpy --docformat google -o ./docs```

in the top level folder of this repo.

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

### CokeBert

You can find the pretrained CokeBert model provided by the authors of the CokeBert paper, which is available from [here](https://drive.google.com/file/d/1Ce7Nq7vJ83l4lOV9SiiN2Kq831z_phsV/view?usp=sharing). 
Then unzip the file, and copy the content of the `DKPLM_BERTbase_2layer` folder into `canlpy/canlpy/pretrained_models/cokebert` 

From [here](https://drive.google.com/file/d/12kOGoaW7yYR_m5SNrlSwH9wVJRAS7HAY/view?usp=sharing), download, run
```shell
tar -xvzf load_data_n.tar.gz
```
and copy the `load_data_n` folder into `canlpy/canlpy/knowledge/cokebert/`

You can find the datasets for the knowledge graph representation used by the authors of the ERNIE paper, download them into this folder from [here](https://drive.google.com/open?id=1HlWw7Q6-dFSm9jNSCh4VaBf1PlGqt9im) and run 

```shell
tar -xvzf data.tar.gz
```
to extract it. 

Then copy `kg_embed` into `canlpy/canlpy/knowledge/cokebert/`

### Examples

Code to use the three models can be found in [examples/](examples/)

### Benchmarks

Benchmarks to compare the models can be found in [experiments/](experiments/)

### Tokens

To use ERNIE and CokeBERT, register your TAGME_TOKEN token in a tokens.py file located in the canlpy/helpers folder

### Acknowledgements

A portion of this code has been adapted from [ERNIE: Enhanced Language Representation with Informative Entities](https://github.com/thunlp/ERNIE), [CokeBERT: Contextual Knowledge Selection and Embedding towards Enhanced Pre-Trained Language Models](https://github.com/thunlp/CokeBERT), [KnowBert: Knowledge Enhanced Contextual Word Representations](https://github.com/allenai/kb)


