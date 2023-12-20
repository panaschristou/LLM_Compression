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
export BNB_CUDA_VERSION=113
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/share/apps/cuda/11.3.1/lib64

cd /home/xc1490/home/projects/hpml/project/bitsandbytes 
python qlora.py \
    --model_name_or_path huggyllama/llama-13b"
