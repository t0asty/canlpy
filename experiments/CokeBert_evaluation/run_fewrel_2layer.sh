#!/bin/bash
#ernie_model
#finetune train
#bert_base => ernie_base
#graphsage

rm output_fewrel_2layer/*

model=../../canlpy/pretrained_models/cokebert

#num_train_epochs 10 -->15
CUDA_VISIBLE_DEVICES=0,1 python3 -W ignore::UserWarning code/run_fewrel_2layer.py   --do_train   --do_lower_case   --data_dir ./data/fewrel/   --ernie_model $model --max_seq_length 256   --train_batch_size 32 --learning_rate 2e-5   --num_train_epochs 16   --output_dir output_fewrel_2layer --loss_scale 128 --K_V_dim 100 --Q_dim 768 --self_att #--graphsage


#eval
CUDA_VISIBLE_DEVICES=0,1 python3 -W ignore::UserWarning code/eval_fewrel_2layer.py   --do_eval   --do_lower_case   --data_dir ./data/fewrel/   --ernie_model $model   --max_seq_length 256  --train_batch_size 32  --learning_rate 2e-5   --num_train_epochs 10   --output_dir output_fewrel_2layer  --loss_scale 128 --K_V_dim 100 --Q_dim 768 --self_att #--graphsage

python3 code/score.py output_fewrel_2layer

