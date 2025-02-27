#!/bin/bash
#SBATCH --job-name=96_pre_joint_s0
#SBATCH --output=sbatch_log/mae96x192_s0_pretrain_joint_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

#SBATCH --nodelist=octopus01,octopus02,octopus03,octopus04,bmicgpu07,bmicgpu08
#SBATCH --cpus-per-task=4
#SBATCH --mem 128GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alexander.eins.qi@gmail.com

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

# nndet_example

# nndet_unpack ${det_data}/Task017_Luna_crop/preprocessed/D3V001_3d/imagesTr 6

nndet_train 022 -o exp.fold=0 \
    train=mae96_plain +augment_cfg.patch_size=[96,192,192] \
    trainer_cfg.gradient_clip_val=0  trainer_cfg.max_num_epochs=60 \
    trainer_cfg.swa_epochs=0 trainer_cfg.warm_iterations=4000  \
    trainer_cfg.amp_backend=None \
    trainer_cfg.precision=32 trainer_cfg.amp_level=None \
    trainer_cfg.scheduler=cosine \
    trainer_cfg.num_val_batches_per_epoch=1000 \
    trainer_cfg.num_train_batches_per_epoch=5000 \
    trainer_cfg.decoder_lr=1e-4 \
    trainer_cfg.initial_lr=1e-5 \
    model_cfg.encoder_kwargs.upsample_func=transpose \
    model_cfg.encoder_kwargs.upsample_stage=direct \
    model_cfg.encoder_kwargs.skip_connection=False \
    model_cfg.encoder_kwargs.pretrained_path=/usr/bmicnas02/data-biwi-01/lung_detection/pretrain_mae/output/pretrain_96_192_192_patch_16/checkpoint-00150_96_192_192.pth \
    --sweep

#train.mode=resume

# mae96x192_submit0_pretrain_joint.sh