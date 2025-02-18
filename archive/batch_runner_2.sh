#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem=400G
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --output=movi-model-check.out
#SBATCH --partition=gpu-h100-80g,gpu-a100-80g,gpu-h200-141g-short

# Exit if a command fails, i.e., if it outputs a non-zero exit status.
set -e

# Get a newer GCC compiler and CUDA compilers

module load gcc/11.4.0 #1oad gcc/11.2.0
module load cuda/12.2.1 #load cuda/11.4.2

# Load environment

module load mamba
source activate slot-attention-jax-locked

# Set extra environment paths so that Jax finds libraries from the conda environment

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX

# Copy training data to the local disk
start=$(date +%s)
# echo " ---- Copying dataset to the tmp folder ---- "
# cp -r movi_e /tmp
# echo " ---- Copied dataset to the tmp folder ---- "
# echo " ---- Elapsed time to copy dataset: $(($end-$start)) seconds ----"
# Run code

srun python -W ignore -m savi.main --config savi/configs/movi/savi++_conditional.py --workdir ckpt_train_plus_plus/

end=$(date +%s)

#python -W ignore -m savi.main --config savi/configs/waymo/waymo_config.py --workdir cktp_waymo/