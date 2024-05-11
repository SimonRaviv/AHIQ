#!/bin/bash

TRAIN_SCRIPT_PATH=/auto/mtrswgwork/simonra/masters/thesis/perceptual_metric/third_party/AHIQ/train.py
RESULTS_PATH=/tmp/ahiq_results
DATASET=$1
SEED=$2
declare -a DATASETS=("LIVE" "CSIQ" "TID2013")
declare -a SEEDS=("42" "84" "126")
if [ -n "$1" ]; then
    DATASETS=("$1")
fi
if [ -n "$2" ]; then
    SEEDS=("$2")
fi
EVALUATION_TYPE="traditional_datasets"
EPOCHS=50
T_MAX=$EPOCHS
LEARNING_RATE=0.0001
WEIGHT_DECAY=0.00001
BATCH_SIZE=8
NUM_WORKERS=8
NUM_AVG_VAL=1
NUM_CROPS=20

# For KADID-10K and PIPAL, the patch size is 8x8:
if [ "$DATASET" == "KADID-10K" ] || [ "$DATASET" == "PIPAL" ]; then
    PATCH_SIZE=8
else
    PATCH_SIZE=16
fi
# EVAL_CENTER_CROP="--eval-center-crop"
EVAL_CENTER_CROP=""

if [ "$EVAL_CENTER_CROP" == "--eval-center-crop" ]; then
    NUM_CROPS=1
fi

mkdir -p $RESULTS_PATH

for dataset in "${DATASETS[@]}"; do
    mkdir -p $RESULTS_PATH/$dataset
    for seed in "${SEEDS[@]}"; do
        results_dir=$RESULTS_PATH/$dataset/$seed
        rm -rf $results_dir
        mkdir -p $results_dir
        python $TRAIN_SCRIPT_PATH \
            --evaluation-type $EVALUATION_TYPE \
            --checkpoints_dir $results_dir \
            --dataset $dataset \
            --name $dataset \
            --seed $seed \
            --n_epoch $EPOCHS \
            --T_max $T_MAX \
            --learning_rate $LEARNING_RATE \
            --weight_decay $WEIGHT_DECAY \
            --batch_size $BATCH_SIZE \
            --num_workers $NUM_WORKERS \
            --num_avg_val $NUM_AVG_VAL \
            --num_crop $NUM_CROPS \
            --patch_size $PATCH_SIZE \
            $EVAL_CENTER_CROP
    done
done
