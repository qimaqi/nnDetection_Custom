#!/bin/bash
#SBATCH --job-name=weightdecay
#SBATCH --output=sbatch_log_new/mae16x224_s0_pretrain_joint_weightdecay%j.out
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

nndet_train 051 -o exp.fold=0 \
    train=mae_plain +augment_cfg.patch_size=[16,224,224] \
    trainer_cfg.freeze_encoder=False \
    trainer_cfg.gradient_clip_val=0  trainer_cfg.max_num_epochs=60 \
    trainer_cfg.swa_epochs=0 trainer_cfg.warm_iterations=4000  \
    trainer_cfg.amp_backend=None \
    trainer_cfg.precision=32 trainer_cfg.amp_level=None \
    trainer_cfg.num_val_batches_per_epoch=1000 \
    trainer_cfg.num_train_batches_per_epoch=5000 \
    trainer_cfg.decoder_lr=1e-4 \
    trainer_cfg.initial_lr=1e-5 \
    trainer_cfg.scheduler=cosine \
    model_cfg.encoder_kwargs.upsample_func=transpose \
    model_cfg.encoder_kwargs.upsample_stage=direct \
    model_cfg.encoder_kwargs.skip_connection=False \
    model_cfg.encoder_kwargs.pretrained_path=/usr/bmicnas02/data-biwi-01/lung_detection/pretrain_mae/output/pretrain_16_224_224/checkpoint-00150.pth \
    trainer_cfg.weight_decay=5.e-4  \
    train.mode=resume \
    --sweep

# train.mode=resume \
#     model_cfg.encoder_kwargs.pretrained_path=/usr/bmicnas02/data-biwi-01/lung_detection/pretrain_mae/output/pretrain_16_224_224/checkpoint-00150.pth \
#    trainer_cfg.weight_decay=5.e-4  \
#mae16x224_submit0_pretrain_joint_weightdecay.sh

#   model_cfg.encoder_kwargs.pretrained_path=/usr/bmicnas02/data-biwi-01/lung_detection/pretrain_mae/output/pretrain_16_224_224/checkpoint-00050.pth \
#trainer_cfg.swa_epochs=0 it runs until the end of the epochs, no problems when resuming
#trainer_cfg.swa_epochs=10 for baseline if we want to run swa for the last 10 epochs and if the max number of epochs is changed