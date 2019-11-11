#!/bin/bash
#SBATCH -J ptbC1e1-sanclAnswersC3e1 -C K80 --partition=gpu-interactive --gres=gpu:1 --time=8:00:00 --output=train.out_ptbC1e1-sanclAnswersC3e1 --error=train.err_ptbC1e1-sanclAnswersC3e1
#SBATCH --mem=20GB
#SBATCH -c 5

module load cudnn/8.0-6.0

/software/tf-old2/bin/python network.py --save_dir saves/ptbC1e1-sanclAnswersC3e1 --model GRNParser --load

