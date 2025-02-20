#!/bin/bash
#SBATCH --time=100:00:00
#SBATCH --mem=500G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --output=peds-waymo-full.out
#SBATCH --partition=gpu-a100-80g,gpu-v100-32g


# Exit if a command fails, i.e., if it outputs a non-zero exit status.
set -e

#ulimit -n 10000

#echo "Open file limit is set to: $(ulimit -n)"

# Get a newer GCC compiler and CUDA compilers

#module load gcc/11.4.0 #1oad gcc/11.2.0
#module load cuda/12.2.1 #load cuda/11.4.2

# Load environment

module load mamba
source activate multi-gpu-slot

# Set extra environment paths so that Jax finds libraries from the conda environment

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX

# Run code> paste back SBATCH --partition=gpu-a100-80g,gpu-h100-80g

python -W ignore -m savi.main --config savi/configs/waymo/waymo_config.py --workdir ckpt_waymo_peds/
