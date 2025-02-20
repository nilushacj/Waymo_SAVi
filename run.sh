#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem=800G
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --output=waymo-model-vehicles.out
#SBATCH --partition=gpu-h200-141g-short

# Exit if a command fails, i.e., if it outputs a non-zero exit status.
set -e

ulimit -n 10000

echo "Open file limit is set to: $(ulimit -n)"

# Get a newer GCC compiler and CUDA compilers

module load gcc/11.4.0 #1oad gcc/11.2.0
module load cuda/12.2.1 #load cuda/11.4.2

# Load environment

module load mamba
source activate slot-attention-jax-locked

# Set extra environment paths so that Jax finds libraries from the conda environment

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX

# Run code> paste back SBATCH --partition=gpu-a100-80g,gpu-h100-80g

srun python -W ignore -m savi.main --config savi/configs/waymo/waymo_vehicles_config.py --workdir ckpt_waymo_vehicles/