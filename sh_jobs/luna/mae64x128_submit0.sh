#!/bin/bash
#SBATCH --job-name=nndet_luna
#SBATCH --output=sbatch_log/mae64_submit_0_fp32_debug_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

#SBATCH --nodelist=bmicgpu08
#SBATCH --cpus-per-task=4
#SBATCH --mem 160GB

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
# export CC=/scratch_net/schusch/qimaqi/install_gcc/bin/gcc-11.3.0
# export CXX=/scratch_net/schusch/qimaqi/install_gcc/bin/g++-11.3.0

export CXX=$CONDA_PREFIX/bin/x86_64-conda_cos6-linux-gnu-c++
export CC=$CONDA_PREFIX/bin/x86_64-conda_cos6-linux-gnu-cc

export det_data="/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_raw"
export det_models="/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_models"
export OMP_NUM_THREADS=1

# nndet_example
# nndet_prep 018
# nndet_unpack ${det_data}/Task018_LunaSWIN/preprocessed/D3V001_3d/imagesTr 6
# nndet_unpack ${det_data}/Task016_Luna/preprocessed/D3V001_3d/imagesTr 6

echo "Job ID: $SLURM_JOBID"
echo "Time: $(date)"


nndet_train 018 -o exp.fold=0 train=mae64_plain  +augment_cfg.patch_size=[64,128,128] trainer_cfg.gradient_clip_val=0  trainer_cfg.max_num_epochs=60 trainer_cfg.swa_epochs=0 trainer_cfg.warm_iterations=4000  trainer_cfg.gradient_clip_val=0 trainer_cfg.amp_backend=None trainer_cfg.precision=32 trainer_cfg.amp_level=None --sweep

# nndet_train 018 -o exp.fold=0 train=mae64 train.mode=resume +augment_cfg.patch_size=[64,128,128] trainer_cfg.gradient_clip_val=0  trainer_cfg.max_num_epochs=60 trainer_cfg.swa_epochs=0 trainer_cfg.warm_iterations=4000  trainer_cfg.gradient_clip_val=0 trainer_cfg.amp_backend=None trainer_cfg.precision=32 trainer_cfg.amp_level=None --sweep


