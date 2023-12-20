echo "Processor: $(lscpu | grep 'Model name' | awk -F ':' '{print $2}' | xargs)"
echo "RAM: $(free -h | grep Mem: | awk '{print $4}')"
echo "GPU: $(nvidia-smi -q | grep 'Product Name')"


prune_ckpt_path='llama_prune/channel_0.2'
tune_ckpt_path='llama_0.2_channel'

echo "[START] - Start Pruning Model"

CUDA_VISIBLE_DEVICES=0 python hf_prune.py --base_model meta-llama/Llama-2-7b-hf --pruning_ratio 0.2 --device cpu  --eval_device cuda --channel_wise --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --test_after_train --taylor param_first --save_model

echo "[FINISH] - Finish Pruning Model"


echo "[START] - Start Tuning"

CUDA_VISIBLE_DEVICES=0 python post_training.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir tune_log/$tune_ckpt_path --wandb_project llama_tune --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64

echo "[FINISH] - Finish Prune and Post-Training."

echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$tune_ckpt_path}/"

echo "You can use the command:"
echo "       python generate.py --model_type tune_prune_LLM --ckpt prune_log/$prune_ckpt_path/pytorch_model.bin --lora_ckpt tune_log/$tune_ckpt_path"
echo "to use the pruned model"

echo "Test speedup of pretrained model"
python test_speedup.py --model_type pretrain --base_model meta-llama/Llama-2-7b-hf
echo "Test ended"

echo "Test Speedup of the pruned model"
python test_speedup.py --model_type pruneLLM --ckpt prune_log/$prune_ckpt_path/pytorch_model.bin	    
echo "Test ended"