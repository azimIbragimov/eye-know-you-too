#!/bin/bash
#SBATCH --job-name=ekyt
#SBATCH --output=ekyt-cnn.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a.ibragimov@ufl.edu
#SBATCH --ntasks=1
#SBATCH --mem=16gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate /blue/cai4104/a.ibragimov/.conda/ekyt-release-new-new
echo hello

cd /blue/cai4104/a.ibragimov/ew/ekyt/eye-know-you-too

# Training EKYT model across every ds value
CUDA_VISIBLE_DEVICES=0 python src/train.py --ds=1 --fold=0
CUDA_VISIBLE_DEVICES=0 python src/train.py --ds=1 --fold=1
CUDA_VISIBLE_DEVICES=0 python src/train.py --ds=1 --fold=2
CUDA_VISIBLE_DEVICES=0 python src/train.py --ds=1 --fold=3

CUDA_VISIBLE_DEVICES=0 python src/train.py --ds=2 --fold=0
CUDA_VISIBLE_DEVICES=0 python src/train.py --ds=2 --fold=1
CUDA_VISIBLE_DEVICES=0 python src/train.py --ds=2 --fold=2
CUDA_VISIBLE_DEVICES=0 python src/train.py --ds=2 --fold=3

CUDA_VISIBLE_DEVICES=0 python src/train.py --ds=4 --fold=0
CUDA_VISIBLE_DEVICES=0 python src/train.py --ds=4 --fold=1
CUDA_VISIBLE_DEVICES=0 python src/train.py --ds=4 --fold=2
CUDA_VISIBLE_DEVICES=0 python src/train.py --ds=4 --fold=3

CUDA_VISIBLE_DEVICES=0 python src/train.py --ds=8 --fold=0
CUDA_VISIBLE_DEVICES=0 python src/train.py --ds=8 --fold=1
CUDA_VISIBLE_DEVICES=0 python src/train.py --ds=8 --fold=2
CUDA_VISIBLE_DEVICES=0 python src/train.py --ds=8 --fold=3

CUDA_VISIBLE_DEVICES=0 python src/train.py --ds=20 --fold=0
CUDA_VISIBLE_DEVICES=0 python src/train.py --ds=20 --fold=1
CUDA_VISIBLE_DEVICES=0 python src/train.py --ds=20 --fold=2
CUDA_VISIBLE_DEVICES=0 python src/train.py --ds=20 --fold=3

CUDA_VISIBLE_DEVICES=0 python src/train.py --ds=32 --fold=0
CUDA_VISIBLE_DEVICES=0 python src/train.py --ds=32 --fold=1
CUDA_VISIBLE_DEVICES=0 python src/train.py --ds=32 --fold=2
CUDA_VISIBLE_DEVICES=0 python src/train.py --ds=32 --fold=3
