# fine-tunning openai-community/gpt2
accelerate launch --num_processes 2 ./ft_llms/llms_finetune.py \
    --output_dir ./ft_llms/openai-community/gpt2/ag_news/refer/ \
    --block_size 128 \
    --eval_steps 100 \
    --save_epochs 100 \
    --log_steps 100 \
    -d ag_news \
    -m openai-community/gpt2 \
    --packing \
    --use_dataset_cache \
    -e 2 -b 4 -lr 5e-5 --gradient_accumulation_steps 1 \
    --train_sta_idx=0 --train_end_idx=4767 --eval_sta_idx=0 --eval_end_idx=538