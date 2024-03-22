#!/bin/bash

TRAIN_SCRIPT_PATH=/auto/mtrswgwork/simonra/masters/thesis/perceptual_metric/third_party/AHIQ/train.py
RESULTS_PATH=/tmp/ahiq_results
DATASET=$1
SEED=$2
CHECKPOINTS_DIR=$3
declare -a DATASETS=("LIVE" "CSIQ" "TID2013")
declare -a SEEDS=("42" "84" "126")
if [ -n "$1" ]; then
    DATASETS=("$1")
fi
# if [ -n "$2" ]; then
#     SEEDS=("$2")
# fi
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
if [ "$TRAIN_DATASETS" == "KADID-10K" ] || [ "$TRAIN_DATASETS" == "PIPAL" ]; then
    PATCH_SIZE=8
else
    PATCH_SIZE=16
fi
# EVAL_CENTER_CROP="--eval-center-crop"
EVAL_CENTER_CROP=""
LOG_EVAL_PREDICTIONS="--log_eval_predictions"

for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        checkpoints_dir=$CHECKPOINTS_DIR/$dataset/$seed
        python $TRAIN_SCRIPT_PATH \
            --evaluation-type $EVALUATION_TYPE \
            --checkpoints_dir $checkpoints_dir \
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
            $EVAL_CENTER_CROP \
            $LOG_EVAL_PREDICTIONS \
            --test
    done
done
