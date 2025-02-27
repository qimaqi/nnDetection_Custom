#!/bin/bash
#SBATCH --job-name=96_luna_s0
#SBATCH --output=sbatch_log/baseline_nndet96x192_s0_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=bmicgpu08
#SBATCH --cpus-per-task=4
#SBATCH --mem 128GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liujiaying423@gmail.com

##SBATCH --account=staff 
##SBATCH --gres=gpu:1
##SBATCH --constraint='titan_xp'

# Load any necessary modules
# srun --account=staff --cpus-per-task=4 --mem 32GB --time 120 --gres=gpu:2 --constraint='titan_xp' --pty bash -i
# srun --account=staff --cpus-per-task=4 --mem 32GB --time 120 --gres=gpu:1 --constraint='titan_xp' --pty bash -i
# srun  --cpus-per-task=4 --mem 32GB --time 120 --gres=gpu:1   --pty bash -i 
# srun  --cpus-per-task=4 --mem 32GB --time 120 --gres=gpu:1 --nodelist=bmicgpu06 --pty bash -i 


source /usr/bmicnas02/data-biwi-01/lung_detection/miniconda3/etc/profile.d/conda.sh
conda activate nndet_venv

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
# export PATH=/scratch_net/schusch/qimaqi/install_gcc:$PATH

export CXX=$CONDA_PREFIX/bin/x86_64-conda_cos6-linux-gnu-c++
export CC=$CONDA_PREFIX/bin/x86_64-conda_cos6-linux-gnu-cc

export det_data="/usr/bmicnas02/data-biwi-01/lung_detection/nnDet_data"
export det_models="/usr/bmicnas02/data-biwi-01/lung_detection/nnDet_models"
export OMP_NUM_THREADS=1

JOB_ID=$(echo $JOB_OUTPUT | awk '{print $4}')

echo "Job ID: $SLURM_JOBID"
echo "Time: $(date)"

# no need
#nndet_prep 019 -o +model_cfg.patch_size=[96,192,192] prep=plan
nndet_train 019 -o exp.fold=0 train=v001 train.mode=resume +augment_cfg.patch_size=[96,192,192] --sweep #train.mode=resume
#baseline96x192_submit0.sh

# nndet_train 018 -o exp.fold=0
# nndet_prep 018
# nndet_unpack ${det_data}/Task018_LunaSWIN/preprocessed/D3V001_3d/imagesTr 6
# nndet_train 016 -o exp.fold=0 train=v001  +augment_cfg.patch_size=[64,128,128] train.mode=resume --sweep 

# nndet_consolidate 016 RetinaUNetV001_D3V001_3d --sweep_boxes --num_folds 1 --shape=64_128_128

# nndet_predict 016 RetinaUNetV001_D3V001_3d --fold 0 --shape=64_128_128


# nndet_eval 018 RetinaUNetV001_D3V001_3d 0 --boxes --analyze_boxes --shape=64_128_128


# echo "Job ID: $SLURM_JOBID"
# echo "Time: $(date)"

# # nndet_train 018 -o exp.fold=0 train=v001 train.mode=resume +augment_cfg.patch_size=[64,128,128] --sweep
# nndet_eval 018 RetinaUNetV001_D3V001_3d 1 --boxes --analyze_boxes --shape=64_128_128


# nndet_train 018 -o exp.fold=2 train=v001  +augment_cfg.patch_size=[64,128,128] --sweep
# nndet_train 018 -o exp.fold=3 train=v001  +augment_cfg.patch_size=[64,128,128] --sweep
# nndet_train 018 -o exp.fold=4 train=v001  +augment_cfg.patch_size=[64,128,128] --sweep

