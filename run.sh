#!/bin/bash

set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate flexi
source ~/.cuda-11.0

stage=0

dataset="evinf"
data_dir="datasets/"
model_dir="trained_models/"

if [ $stage -le 0 ]; then
    for seed in 5 10 15; do
        python finetune_on_ful.py --dataset $dataset \
                                  --model_dir $model_dir \
                                  --data_dir $data_dir \
                                  --seed $seed 
    done

    python finetune_on_ful.py --dataset $dataset \
                              --model_dir $model_dir \
                              --data_dir $data_dir \
                              --seed $seed  \
                              --evaluate_models
fi

divergence="jsd"
extracted_rationale_dir="extracted_rationales/"

if [ $stage -le 1 ]; then
    python extract_rationales.py --dataset $dataset  \
                                 --model_dir $model_dir \
                                 --data_dir $data_dir \
                                 --extracted_rationale_dir $extracted_rationale_dir \
                                 --extract_double \
                                 --divergence $divergence
fi

thresh="topk"
evaluation_dir="faithfulness_metrics/"

if [ $stage -le 2 ]; then
    python evaluate_masked.py --dataset $dataset \
                              --model_dir $model_dir \
                              --extracted_rationale_dir $extracted_rationale_dir \
                              --data_dir $data_dir \
                              --evaluation_dir $evaluation_dir\
                              --thresholder $thresh
fi
