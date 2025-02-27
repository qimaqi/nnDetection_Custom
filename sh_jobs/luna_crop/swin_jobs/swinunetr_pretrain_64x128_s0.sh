#!/bin/bash
#SBATCH --job-name=sw_pre_jo_64
#SBATCH --output=sbatch_log/swin_64_pretrain_joint_epoch55_s0_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

#SBATCH --nodelist=octopus01,octopus02,octopus03,octopus04,bmicgpu07,bmicgpu08,bmicgpu09
#SBATCH --cpus-per-task=4
#SBATCH --mem 128GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liujiaying423@gmail.com

##SBATCH --gres=gpu:1
##SBATCH --constraint='titan_xp'

# Load any necessary modules
# srun --account=staff --cpus-per-task=4 --mem 32GB --time 120 --gres=gpu:2 --constraint='titan_xp' --pty bash -i
# srun --account=staff --cpus-per-task=4 --mem 32GB --time 120 --gres=gpu:1 --constraint='titan_xp' --pty bash -i
# srun  --cpus-per-task=4 --mem 32GB --time 120 --gres=gpu:1   --pty bash -i 
# srun  --cpus-per-task=4 --mem 32GB --time 120 --gres=gpu:1 --nodelist=bmicgpu06 --pty bash -i 


# source /usr/bmicnas02/data-biwi-01/lung_detection/miniconda3/etc/profile.d/conda.sh
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

# nndet_example
#For luna costum 16*224*224

# nndet_unpack ${det_data}/Task017_Luna_crop/preprocessed/D3V001_3d/imagesTr 6

# nndet_train 018 -o exp.fold=0 train=swinunetr train.mode=resume +augment_cfg.patch_size=[64,128,128]  trainer_cfg.gradient_clip_val=0  trainer_cfg.max_num_epochs=60 trainer_cfg.swa_epochs=0 trainer_cfg.warm_iterations=4000  trainer_cfg.gradient_clip_val=0 trainer_cfg.amp_backend=None trainer_cfg.precision=32 trainer_cfg.amp_level=None --sweep


nndet_train 021 -o exp.fold=0 train=swinunetr \
    +trainer_cfg.freeze_encoder=False \
    +augment_cfg.patch_size=[64,128,128]  \
    trainer_cfg.gradient_clip_val=0  \
    trainer_cfg.max_num_epochs=60 \
    trainer_cfg.swa_epochs=0 \
    trainer_cfg.warm_iterations=4000  \
    trainer_cfg.gradient_clip_val=0 \
    trainer_cfg.amp_backend=None \
    trainer_cfg.precision=32 \
    trainer_cfg.amp_level=None \
    trainer_cfg.num_val_batches_per_epoch=1000 \
    trainer_cfg.num_train_batches_per_epoch=5000 \
    model_cfg.encoder_kwargs.pretrained_path=/usr/bmicnas02/data-biwi-01/lung_detection/pretrain_swin/runs/64_128_128/model_epoch_55.pt \
    trainer_cfg.scheduler=poly \
    trainer_cfg.optimizer=adamw \
    train.mode=resume \
    --sweep

#train.mode=resume \
#swinunetr_pretrain_64x128_s0.sh

