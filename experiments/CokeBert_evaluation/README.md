# CokeBert Experiments

First, make sure that you have the pretrained CokeBert model in `canlpy/canlpy/pretrained_models/cokebert`.  
If that is not the case, you find the pretrained CokeBert model provided by the authors of the CokeBert paper, which is available from [here](https://drive.google.com/file/d/1poBynySbIosIYHaHfqnHKQEqoMmm1h5K/view?usp=sharing). 
Then run
```shell
tar -xvzf cokebert.tar.gz
```
and copy the content of the folder into `canlpy/canlpy/pretrained_models/cokebert` 

From the same file, copy the `load_data_n` folder into `./data/`

You can find the datasets for fine-tuning and the knowledge graph representation used by the authors of the ERNIE paper, download them into this folder from [here](https://drive.google.com/open?id=1HlWw7Q6-dFSm9jNSCh4VaBf1PlGqt9im) and run 

```shell
tar -xvzf data.tar.gz
```
to extract it. 

Then copy all contained folders into `./data/`

To fine-tune then run from this folder

**FewRel:**

```bash
bash run_fewrel_2layer.sh
````

**TACRED:**

```bash
bash run_tacred_2layer.sh
```

**OpenEntity:**

```bash
bash run_open_2layer.sh
```

You can then find all results in the respective output folder.
