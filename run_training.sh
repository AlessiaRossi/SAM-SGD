#!/bin/bash
#SBATCH -A NAISS2025-5-98 -p alvis
#SBATCH --job-name=SAM_SGD_Training
#SBATCH --output=output_%j.txt
#SBATCH --ntasks=1
#SBATCH -t 0-75:00:0
#SBATCH --gpus-per-node=A40:1

# Carica i moduli necessari
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Attiva il virtual environment
source .venv/bin/activate

# Esegui il tuo script Python
python train.py \
    --batch_size 128 \
    --depth 2 \
    --epochs 10 \
    --learning_rate 0.1 \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --rho 0.05 \
    --lambda_ 0.7 \
    --lambda_range 0,1,0.2 \
    --dataset cifar10 \
    --optimize_lambda
 