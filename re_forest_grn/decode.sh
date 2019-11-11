#!/bin/bash
#SBATCH --partition=gpu-interactive --gres=gpu:1 --time=1:00:00 --output=decode.out --error=decode.err
#SBATCH --mem=10GB
#SBATCH -c 6

module load anaconda/5.1.0 cudnn/9.0-7

testset=dev

start=`date +%s`
python G2S_evaluator.py --model_prefix logs/G2S.$1 \
        --in_path data/$testset\.json \
        --in_dep_path data/$testset\.json_$2 \
        --out_path logs/res_$testset\_$1\.json

end=`date +%s`
runtime=$((end-start))
echo $runtime
