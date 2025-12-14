python ./transformers/examples/pytorch/text-classification/run_classification_new.py \
    --model_name_or_path distilbert/distilbert-base-uncased \
    --train_file /home/zjq/nlp-project/data/repair_nli_v2/train.csv \
    --validation_file /home/zjq/nlp-project/data/repair_nli_v2/valid.csv \
    --test_file /home/zjq/nlp-project/data/repair_nli_v2/test.csv \
    --shuffle_train_dataset \
    --metric_name accuracy \
    --label_smoothing_factor 0.1 \
    --text_column_names "text" \
    --label_column_name "label" \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 512 \
    --early_stopping_patience 3 \
    --per_device_train_batch_size 64 \
    --learning_rate 2e-5 \
    --num_train_epochs 4 \
    --output_dir ./output/1214/1214-clean-head-early-real-4epoch/ \
    --trust_remote_code True \
    --report_to wandb \
    --run_name 1214-clean-head-early-real-4epoch \
    --dataloader_num_workers 16 \
    --pad_to_max_length False \
    --warmup_steps 1000 \
    --load_best_model_at_end True \
    --save_strategy steps \
    --save_steps 2000 \
    --eval_strategy steps \
    --eval_steps 2000 \
    --metric_for_best_model accuracy \
    --lr_scheduler_type cosine


# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./transformers/examples/pytorch/text-classification/run_classification.py \
#     --model_name_or_path distilroberta-base \
#     --train_file /home/zjq/nlp-project/data/ori/train.csv \
#     --validation_file /home/zjq/nlp-project/data/ori/valid.csv \
#     --test_file /home/zjq/nlp-project/data/ori/test.csv \
#     --shuffle_train_dataset \
#     --metric_name accuracy \
#     --text_column_names "text" \
#     --label_column_name "label" \
#     --early_stopping_patience 3 \
#     --do_train \
#     --do_eval \
#     --do_predict \
#     --max_seq_length 512 \
#     --per_device_train_batch_size 64 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 4 \
#     --output_dir ./output/distilbert-base-dataset-3epoch/ \
#     --trust_remote_code True \
#     --report_to wandb \
#     --run_name amazon_reviews_1213_early \
#     --dataloader_num_workers 16 \
#     --pad_to_max_length False \
#     --clean_text True \
#     --lowercase_text True \
#     --eval_strategy steps \
#     --eval_steps 2000 \
#     --save_strategy steps \
#     --save_steps 2000