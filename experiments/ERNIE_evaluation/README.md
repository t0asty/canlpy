# ERNIE Experiments

First make sure that you have the pretrained ERNIE model in `canlpy/canlpy/pretrained_models/ernie`.  
If that is not the case, you find the pretrained ERNIE model provided by the authors of the ERNIE paper, which is available from [here](https://drive.google.com/file/d/1Hdp_iqsF3xjFcWSRvklC5ppvvd2C0qim/view?usp=sharing). Then run

```shell
tar -xvzf ernie_base.tar.gz
```
to extract it into `canlpy/canlpy/pretrained_models/ernie` 

You can find the datasets for fine-tuning and the knowledge graph representation used by the authors of the ERNIE paper, download them into this folder from [here](https://drive.google.com/open?id=1HlWw7Q6-dFSm9jNSCh4VaBf1PlGqt9im) and run 

```shell
tar -xvzf data.tar.gz
```
to extract it. Move the content of `kg_embed` into `canlpy/canlpy/knowledge/ernie/`, all other folders stay in `data`. 

To fine-tune then run from this folder

**FewRel:**

```bash
python3 code/run_fewrel.py   --do_train   --do_lower_case   --data_dir data/fewrel/   --ernie_model ../../canlpy/pretrained_models/ernie   --max_seq_length 256   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 10   --output_dir output_fewrel  --loss_scale 128
# evaluate
python3 code/eval_fewrel.py   --do_eval   --do_lower_case   --data_dir data/fewrel/   --ernie_model ../../canlpy/pretrained_models/ernie   --max_seq_length 256   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 10   --output_dir output_fewrel  --loss_scale 128
````

**TACRED:**

```bash
python3 code/run_tacred.py   --do_train   --do_lower_case   --data_dir data/tacred   --ernie_model ../../canlpy/pretrained_models/ernie   --max_seq_length 256   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 4.0   --output_dir output_tacred   --loss_scale 128 --threshold 0.4
# evaluate
python3 code/eval_tacred.py   --do_eval   --do_lower_case   --data_dir data/tacred   --ernie_model ../../canlpy/pretrained_models/ernie   --max_seq_length 256   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 4.0   --output_dir output_tacred   --loss_scale 128 --threshold 0.4
```

**OpenEntity:**

```bash
python3 code/run_typing.py    --do_train   --do_lower_case   --data_dir data/OpenEntity   --ernie_model ../../canlpy/pretrained_models/ernie   --max_seq_length 128   --train_batch_size 16   --learning_rate 2e-5   --num_train_epochs 10.0   --output_dir output_open --threshold 0.3 --loss_scale 128
# evaluate
python3 code/eval_typing.py   --do_eval   --do_lower_case   --data_dir data/OpenEntity   --ernie_model ../../canlpy/pretrained_models/ernie   --max_seq_length 128   --train_batch_size 16   --learning_rate 2e-5   --num_train_epochs 10.0   --output_dir output_open --threshold 0.3 --loss_scale 128
```

The script for the evaluation of relation classification just gives the accuracy score. For the macro/micro metrics, you should use `code/score.py` which is from [tacred repo](<https://github.com/yuhaozhang/tacred-relation>).

```shell
python3 code/score.py gold_file pred_file
```

You can find `gold_file` and `pred_file` on each checkpoint in the output folder (`--output_dir`).
