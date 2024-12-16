#!/bin/bash
#SBATCH --job-name=nndet_luna
#SBATCH --output=sbatch_log/swinunter64x128_s1_debug_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

## SBATCH --account=staff
#SBATCH --nodelist=bmicgpu08
#SBATCH --cpus-per-task=4
#SBATCH --mem 160GB


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
# nndet_prep 018 -o exp.fold=0 train=v001 overwrite=True model_cfg.encoder_kwargs={'plan_size':[64,128,128]}

echo "Job ID: $SLURM_JOBID"
echo "Time: $(date)"

nndet_train 018 -o exp.fold=0 train=swinunetr +augment_cfg.patch_size=[64,128,128] trainer_cfg.gradient_clip_val=0 trainer_cfg.amp_backend=None trainer_cfg.precision=32 trainer_cfg.amp_level=None trainer_cfg.max_num_epochs=5 trainer_cfg.swa_epochs=0 trainer_cfg.warm_iterations=0 --sweep 

# train.mode=resume 

# nndet_predict 018 SwinUnetrV001_D3V001_3d  --fold 0 --num_models 1 --shape 64_128_128 --no_preprocess --test_split


# nndet_eval 018 SwinUnetrV001_D3V001_3d 0 --boxes --analyze_boxes --shape=64_128_128

