# If it's full parameter training, use `--model xxx` instead of `--adapters xxx`.
# If you are using the validation set for inference, add the parameter `--load_data_args true`.
#--model_type mistral、llama3_1、qwen3、glm4_0414

#
#CUDA_VISIBLE_DEVICES=2 swift infer \
#    --model /home/dataica401/PersonalFile/kanghankun/lijinliang/LLMS/Mistral/Mistral-7B-Instruct-v0.3 \
#    --model_type mistral \
#    --adapters /home/dataica401/PersonalFile/kanghankun/piqiankun/ms-swift/output/mistral3-7b-lora/v1-20260106-212359/checkpoint-21 \
#    --stream true \
#    --infer_backend vllm \
#    --max_batch_size 96 \
#    --val_dataset '/home/dataica401/PersonalFile/kanghankun/piqiankun/ms-swift/swift/llm/dataset/data/sft_wvs_test.json' \
#    --result_path /home/dataica401/PersonalFile/kanghankun/piqiankun/ms-swift/swift/llm/dataset/data/saves/mistral3-7b-中6-中7\
#    --max_new_tokens 64


#    --adapters /home/dataica401/PersonalFile/kanghankun/piqiankun/ms-swift/output/美国/llama3-8b-lora-不加重大事件/v0-20260119-190155/checkpoint-10 \

CUDA_VISIBLE_DEVICES=1  swift infer \
    --model /mnt/data/LLMS/Qwen/Qwen3-14B \
    --model_type qwen3 \
    --stream true \
    --infer_backend vllm \
    --max_batch_size 96 \
    --val_dataset '/mnt/data/piqiankun/project/ms-swift/swift/llm/dataset/data/直接推理.json' \
    --result_path /mnt/data/piqiankun/project/ms-swift/swift/llm/dataset/data/saves/qwen3-14b-中7\
    --max_new_tokens 256


#CUDA_VISIBLE_DEVICES=3  swift infer \
#    --model /home/dataica401/PersonalFile/kanghankun/lijinliang/LLMS/GLM/GLM-4-9B-0414  \
#    --model_type glm4_0414 \
#    --stream true \
#    --infer_backend vllm \
#    --max_batch_size 96 \
#    --val_dataset '/home/dataica401/PersonalFile/kanghankun/piqiankun/ms-swift/swift/llm/dataset/data/中6-中7plus-question.json' \
#    --result_path /home/dataica401/PersonalFile/kanghankun/piqiankun/ms-swift/swift/llm/dataset/data/saves/qwen3-question-14b-中6-中7\
#    --max_new_tokens 4096