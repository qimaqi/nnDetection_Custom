#!/bin/bash
#SBATCH --job-name=nndet_luna
#SBATCH --output=sbatch_log/baseline_nndet_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

#SBATCH --nodelist=bmicgpu06
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16GB
##SBATCH --mem 32GB

##SBATCH --account=staff 
##SBATCH --gres=gpu:1
##SBATCH --constraint='titan_xp'

# Load any necessary modules
# srun --account=staff --cpus-per-task=4 --mem 32GB --time 120 --gres=gpu:2 --constraint='titan_xp' --pty bash -i
# srun --account=staff --cpus-per-task=4 --mem 32GB --time 120 --gres=gpu:1 --constraint='titan_xp' --pty bash -i
# srun  --cpus-per-task=4 --mem 32GB --time 120 --gres=gpu:1   --pty bash -i 
# srun  --cpus-per-task=4 --mem 32GB --time 120 --gres=gpu:1 --nodelist=bmicgpu06 --pty bash -i 


source /scratch_net/schusch/qimaqi/miniconda3/etc/profile.d/conda.sh
conda activate nndet_swin

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export PATH=/scratch_net/schusch/qimaqi/install_gcc:$PATH

export CXX=$CONDA_PREFIX/bin/x86_64-conda_cos6-linux-gnu-c++
export CC=$CONDA_PREFIX/bin/x86_64-conda_cos6-linux-gnu-cc

export det_data="/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_raw"
export det_models="/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_models"
export OMP_NUM_THREADS=1

# nndet_example
nndet_prep 018 -o train=swinunetr 

nndet_unpack ${det_data}/Task018_LunaSWIN/preprocessed/D3V001_3d/imagesTr 6

# nndet_train 017 -o exp.fold=1 train=swinunetr
# [ 80 192 160] patch size as default
# we use [64, 128, 128] as new standard patch size for experiments
# first check how the slices look like
# (1, 240, 441, 441) it the shape 
# (1, 285, 427, 427)
# nndet_train 018 -o exp.fold=1 train=v001
# nndet_train 018 -o exp.fold=0 train=swinunetr