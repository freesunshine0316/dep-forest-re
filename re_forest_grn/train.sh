#!/bin/bash
#SBATCH -J first100_1k --partition=gpu --gres=gpu:1 --time=8:00:00 --output=train.out_first100_1k --error=train.err_first100_1k
#SBATCH --mem=20GB
#SBATCH -c 6

module load anaconda/5.1.0 cudnn/9.0-7

python G2S_trainer.py --config_path logs/G2S.first100_1k.config.json

