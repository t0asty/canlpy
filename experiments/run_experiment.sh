#!/bin/bash
while getopts m:d: flag
do
        case "${flag}" in
                    m) model=${OPTARG};;
                    d) dataset=${OPTARG};;
        esac
done

model="$(tr [A-Z] [a-z] <<< "$model")"
dataset="$(tr [A-Z] [a-z] <<< "$dataset")"

echo_help () {
    echo ""
    echo "intended use: "
    echo "-m: model to get results for (should be one of 'ernie', 'cokebert')"
    echo "-d: dataset to get results for (should be one of 'fewrel', 'tacred', 'openentity')"
    echo ""
    echo "e.g. bash run_experiment.sh -m ernie -d fewrel"
}

implementedModels=("ernie" "cokebert")
availableDatasets=("fewrel" "tacred" "openentity")

if !([[ " ${implementedModels[*]} " =~ " ${model} " ]];) then
    echo_help
else 
    if !([[ " ${availableDatasets[*]} " =~ " ${dataset} " ]];) then
        echo_help
    fi
fi

if [ "$model" = "ernie" ]; then
    cd ./ERNIE_evaluation

    if [ "$dataset" = "fewrel" ]; then
        python3 code/run_fewrel.py   --do_train   --do_lower_case   --data_dir data/fewrel/   --ernie_model ernie_base   --max_seq_length 256   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 10   --output_dir output_fewrel   --loss_scale 128
        python3 code/eval_fewrel.py   --do_eval   --do_lower_case   --data_dir data/fewrel/   --ernie_model ernie_base   --max_seq_length 256   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 10   --output_dir output_fewrel   --loss_scale 128
    fi
    if [ "$dataset" = "tacred" ]; then
        python3 code/run_tacred.py   --do_train   --do_lower_case   --data_dir data/tacred   --ernie_model ernie_base   --max_seq_length 256   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 4.0   --output_dir output_tacred   --loss_scale 128 --threshold 0.4
        python3 code/eval_tacred.py   --do_eval   --do_lower_case   --data_dir data/tacred   --ernie_model ernie_base   --max_seq_length 256   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 4.0   --output_dir output_tacred   --loss_scale 128 --threshold 0.4
    fi
    if [ "$dataset" = "openentity" ]; then
        python3 code/run_typing.py    --do_train   --do_lower_case   --data_dir data/OpenEntity   --ernie_model ernie_base   --max_seq_length 128   --train_batch_size 16   --learning_rate 2e-5   --num_train_epochs 10.0   --output_dir output_open --threshold 0.3 --loss_scale 128
        python3 code/eval_typing.py   --do_eval   --do_lower_case   --data_dir data/OpenEntity   --ernie_model ernie_base   --max_seq_length 128   --train_batch_size 16   --learning_rate 2e-5   --num_train_epochs 10.0   --output_dir output_open --threshold 0.3 --loss_scale 128
    fi
fi

if [ "$model" = "cokebert" ]; then
    cd ./CokeBert_evaluation
    
    if [ "$dataset" = "fewrel" ]; then
        bash run_fewrel_2layer.sh
    fi
    if [ "$dataset" = "tacred" ]; then
        bash run_tacred_2layer.sh
    fi
    if [ "$dataset" = "openentity" ]; then
        bash run_open_2layer.sh
    fi
fi