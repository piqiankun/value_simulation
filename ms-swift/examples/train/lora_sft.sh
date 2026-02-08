# 22GB
# qwen3: https://github.com/modelscope/ms-swift/blob/main/examples/train/think_model/qwen3_demo1.sh
CUDA_VISIBLE_DEVICES=2 \
swift sft \
    --model /home/dataica401/PersonalFile/kanghankun/lijinliang/LLMS/Llama/Meta-Llama-3.1-8B-Instruct \
    --model_type llama3_1 \
    --train_type lora \
    --dataset '/home/dataica401/PersonalFile/kanghankun/piqiankun/ms-swift/swift/llm/dataset/data/qwen3-14b-微调.json' \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 96 \
    --learning_rate 3e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir /home/dataica401/PersonalFile/kanghankun/piqiankun/ms-swift/output/llama3-8b-lora \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.08 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
