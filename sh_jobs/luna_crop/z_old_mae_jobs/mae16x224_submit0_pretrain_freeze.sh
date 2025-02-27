#!/bin/bash
#SBATCH --job-name=16_pre_fre_s0
#SBATCH --output=sbatch_log/mae16x224_s0_pretrain_freeze_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

#SBATCH --nodelist=octopus01,octopus02,octopus03,octopus04,bmicgpu07,bmicgpu08,bmicgpu09
#SBATCH --cpus-per-task=4
#SBATCH --mem 128GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alexander.eins.qi@gmail.com

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

nndet_train 023 -o exp.fold=0 \
    train=mae_plain +augment_cfg.patch_size=[16,224,224] \
    trainer_cfg.gradient_clip_val=0  trainer_cfg.max_num_epochs=60 \
    trainer_cfg.swa_epochs=0 trainer_cfg.warm_iterations=4000  \
    trainer_cfg.amp_backend=None \
    trainer_cfg.freeze_encoder=True \
    trainer_cfg.num_val_batches_per_epoch=1000 \
    trainer_cfg.num_train_batches_per_epoch=5000 \
    trainer_cfg.precision=32 trainer_cfg.amp_level=None \
    trainer_cfg.scheduler=poly \
    train.mode=resume \
    model_cfg.encoder_kwargs.pretrained_path=/usr/bmicnas02/data-biwi-01/lung_detection/pretrain_mae/output/pretrain_16_224_224/checkpoint-last.pth --sweep

# train.mode=resume \

# mae16x224_submit0_pretrain_freeze

#trainer_cfg.swa_epochs=0 it runs until the end of the epochs, no problems when resuming
#trainer_cfg.swa_epochs=10 for baseline if we want to run swa for the last 10 epochs and if the max number of epochs is changed