#!/bin/bash

#SBATCH --job-name=qlora
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=36GB
#SBATCH --time=47:59:00
#SBATCH --gres=gpu:a100:1
module purge
module load cudnn/8.6.0.163-cuda11
module load cuda/11.3.1
overlay_ext3=/home/xc1490/home/apps/llm2/overlay-15GB-500K.ext3
singularity exec --nv \
    --overlay ${overlay_ext3}:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
cd /scratch/xc1490/projects/hpml/project/bitsandbytes
export BNB_CUDA_VERSION=113
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/share/apps/cuda/11.3.1/lib64
python qlora.py \
    --model_name_or_path '/scratch/xc1490/projects/hpml/project/sparsegpt/output/llama_0.1' --tokenizer huggyllama/llama-13b --use_auth --output_dir ./output/sparsegpt_0.1_quant --logging_steps 10 --save_strategy steps --data_seed 42 --save_steps 500 --save_total_limit 40 --evaluation_strategy steps --eval_dataset_size 1024 --max_eval_samples 1000 --per_device_eval_batch_size 1 --max_new_tokens 32 --dataloader_num_workers 1 --group_by_length --logging_strategy steps --remove_unused_columns False --do_train --do_eval --do_mmlu_eval --lora_r 64 --lora_alpha 16 --lora_modules all --double_quant --quant_type nf4 --bf16 --bits 4 --warmup_ratio 0.03 --lr_scheduler_type constant --gradient_checkpointing --dataset oasst1 --source_max_len 16 --target_max_len 512 --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --max_steps 1875 --eval_steps 187 --learning_rate 0.0002 --adam_beta2 0.999 --max_grad_norm 0.3 --lora_dropout 0.1 --weight_decay 0.0 --seed 0
python qlora.py \
    --model_name_or_path '/scratch/xc1490/projects/hpml/project/sparsegpt/output/llama_0.2' --tokenizer huggyllama/llama-13b --use_auth --output_dir ./output/sparsegpt_0.2_quant --logging_steps 10 --save_strategy steps --data_seed 42 --save_steps 500 --save_total_limit 40 --evaluation_strategy steps --eval_dataset_size 1024 --max_eval_samples 1000 --per_device_eval_batch_size 1 --max_new_tokens 32 --dataloader_num_workers 1 --group_by_length --logging_strategy steps --remove_unused_columns False --do_train --do_eval --do_mmlu_eval --lora_r 64 --lora_alpha 16 --lora_modules all --double_quant --quant_type nf4 --bf16 --bits 4 --warmup_ratio 0.03 --lr_scheduler_type constant --gradient_checkpointing --dataset oasst1 --source_max_len 16 --target_max_len 512 --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --max_steps 1875 --eval_steps 187 --learning_rate 0.0002 --adam_beta2 0.999 --max_grad_norm 0.3 --lora_dropout 0.1 --weight_decay 0.0 --seed 0
python qlora.py \
    --model_name_or_path '/scratch/xc1490/projects/hpml/project/sparsegpt/output/llama_0.3' --tokenizer huggyllama/llama-13b --use_auth --output_dir ./output/sparsegpt_0.3_quant --logging_steps 10 --save_strategy steps --data_seed 42 --save_steps 500 --save_total_limit 40 --evaluation_strategy steps --eval_dataset_size 1024 --max_eval_samples 1000 --per_device_eval_batch_size 1 --max_new_tokens 32 --dataloader_num_workers 1 --group_by_length --logging_strategy steps --remove_unused_columns False --do_train --do_eval --do_mmlu_eval --lora_r 64 --lora_alpha 16 --lora_modules all --double_quant --quant_type nf4 --bf16 --bits 4 --warmup_ratio 0.03 --lr_scheduler_type constant --gradient_checkpointing --dataset oasst1 --source_max_len 16 --target_max_len 512 --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --max_steps 1875 --eval_steps 187 --learning_rate 0.0002 --adam_beta2 0.999 --max_grad_norm 0.3 --lora_dropout 0.1 --weight_decay 0.0 --seed 0
python qlora.py \
    --model_name_or_path '/scratch/xc1490/projects/hpml/project/sparsegpt/output/llama_0.4' --tokenizer huggyllama/llama-13b --use_auth --output_dir ./output/sparsegpt_0.4_quant --logging_steps 10 --save_strategy steps --data_seed 42 --save_steps 500 --save_total_limit 40 --evaluation_strategy steps --eval_dataset_size 1024 --max_eval_samples 1000 --per_device_eval_batch_size 1 --max_new_tokens 32 --dataloader_num_workers 1 --group_by_length --logging_strategy steps --remove_unused_columns False --do_train --do_eval --do_mmlu_eval --lora_r 64 --lora_alpha 16 --lora_modules all --double_quant --quant_type nf4 --bf16 --bits 4 --warmup_ratio 0.03 --lr_scheduler_type constant --gradient_checkpointing --dataset oasst1 --source_max_len 16 --target_max_len 512 --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --max_steps 1875 --eval_steps 187 --learning_rate 0.0002 --adam_beta2 0.999 --max_grad_norm 0.3 --lora_dropout 0.1 --weight_decay 0.0 --seed 0
python qlora.py \
    --model_name_or_path '/scratch/xc1490/projects/hpml/project/sparsegpt/output/llama_0.5' --tokenizer huggyllama/llama-13b --use_auth --output_dir ./output/sparsegpt_0.5_quant --logging_steps 10 --save_strategy steps --data_seed 42 --save_steps 500 --save_total_limit 40 --evaluation_strategy steps --eval_dataset_size 1024 --max_eval_samples 1000 --per_device_eval_batch_size 1 --max_new_tokens 32 --dataloader_num_workers 1 --group_by_length --logging_strategy steps --remove_unused_columns False --do_train --do_eval --do_mmlu_eval --lora_r 64 --lora_alpha 16 --lora_modules all --double_quant --quant_type nf4 --bf16 --bits 4 --warmup_ratio 0.03 --lr_scheduler_type constant --gradient_checkpointing --dataset oasst1 --source_max_len 16 --target_max_len 512 --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --max_steps 1875 --eval_steps 187 --learning_rate 0.0002 --adam_beta2 0.999 --max_grad_norm 0.3 --lora_dropout 0.1 --weight_decay 0.0 --seed 0
python qlora.py \
    --model_name_or_path '/scratch/xc1490/projects/hpml/project/sparsegpt/output/llama_0.6' --tokenizer huggyllama/llama-13b --use_auth --output_dir ./output/sparsegpt_0.6_quant --logging_steps 10 --save_strategy steps --data_seed 42 --save_steps 500 --save_total_limit 40 --evaluation_strategy steps --eval_dataset_size 1024 --max_eval_samples 1000 --per_device_eval_batch_size 1 --max_new_tokens 32 --dataloader_num_workers 1 --group_by_length --logging_strategy steps --remove_unused_columns False --do_train --do_eval --do_mmlu_eval --lora_r 64 --lora_alpha 16 --lora_modules all --double_quant --quant_type nf4 --bf16 --bits 4 --warmup_ratio 0.03 --lr_scheduler_type constant --gradient_checkpointing --dataset oasst1 --source_max_len 16 --target_max_len 512 --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --max_steps 1875 --eval_steps 187 --learning_rate 0.0002 --adam_beta2 0.999 --max_grad_norm 0.3 --lora_dropout 0.1 --weight_decay 0.0 --seed 0"

