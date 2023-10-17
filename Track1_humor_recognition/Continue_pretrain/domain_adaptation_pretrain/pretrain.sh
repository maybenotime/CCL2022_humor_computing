MODEL_NAME=/aliyun-06/share_v2/yangshiping/projects/humor_computation/pretrain_data/checkpoint
CHECKPOINT=../final_checkpoint
TRAIN_FILE=../clean_data/train.txt
EVAL_FILE=../clean_data/dev.txt
LR=3e-8

python continue_pretrain.py \
    --model_name_or_path $MODEL_NAME \
    --model_type bert \
    --train_data_file $TRAIN_FILE \
    --eval_data_file $EVAL_FILE \
    --per_device_train_batch_size 32 \
    --do_train \
    --learning_rate $LR\
    --do_eval \
    --mlm \
    --line_by_line \
    --output_dir $CHECKPOINT \
    --overwrite_output_dir \
    --block_size 400 \
    --save_total_limit 5 \
    --num_train_epoch 90 \
    --load_best_model_at_end
