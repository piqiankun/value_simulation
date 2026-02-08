NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model /mnt/data/LLMS/Qwen/Qwen3-8B \
    --infer_backend vllm \
    --val_dataset swift/中国第7轮_1 \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 8192 \
    --vllm_tensor_parallel_size 2 \
    --max_new_tokens 2048 \
    --write_batch_size 1000
