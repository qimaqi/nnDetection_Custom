#!/bin/bash
#SBATCH --job-name=nndet_luna
#SBATCH --output=sbatch_log/baseline_nndet_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

#SBATCH --nodelist=bmicgpu08
#SBATCH --cpus-per-task=4
#SBATCH --mem 96GB

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
# export CC=/scratch_net/schusch/qimaqi/install_gcc/bin/gcc-11.3.0
# export CXX=/scratch_net/schusch/qimaqi/install_gcc/bin/g++-11.3.0

export CXX=$CONDA_PREFIX/bin/x86_64-conda_cos6-linux-gnu-c++
export CC=$CONDA_PREFIX/bin/x86_64-conda_cos6-linux-gnu-cc

export det_data="/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_raw"
export det_models="/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnDet_models"
export OMP_NUM_THREADS=1

# nndet_example
# nndet_prep 016 
# nndet_unpack ${det_data}/Task000D3_Example/preprocessed/D3V001_3d/imagesTr 6
# nndet_unpack ${det_data}/Task016_Luna/preprocessed/D3V001_3d/imagesTr 6

# nndet_train 016 --sweep
# nndet_predict 016 RetinaUNetV001_D3V001_3d --fold -1
# nndet_consolidate 016 RetinaUNetV001_D3V001_3d --sweep_boxes
# nndet_eval 016 RetinaUNetV001_D3V001_3d 0 --boxes --analyze_boxes

# nndet_sweep 016 RetinaUNetV001_D3V001_3d 0

nndet_train 016 -o exp.fold=0 train=v001


# ====================================
# INFO Architecture overwrites: {} Anchor overwrites: {}
# INFO Building architecture according to plan of RetinaUNetV001
# INFO Start channels: 32; head channels: 128; fpn channels: 128
# INFO Discarding anchor generator kwargs {'stride': 1}
# INFO Building:: encoder Encoder: {} 
# INFO Building:: decoder UFPNModular: {'min_out_channels': 8, 'upsampling_mode': 'transpose', 'num_lateral': 1, 'norm_lateral': False, 'activation_lateral': False, 'num_out': 1, 'norm_out': False, 'activation_out': False}
# INFO Running ATSS Matching with num_candidates=4 and center_in_gt False.
# INFO Building:: classifier BCECLassifier: {'num_convs': 1, 'norm_channels_per_group': 16, 'norm_affine': True, 'reduction': 'mean', 'loss_weight': 1.0, 'prior_prob': 0.01}
# INFO Init classifier weights: prior prob 0.01
# INFO Building:: regressor GIoURegressor: {'num_convs': 1, 'norm_channels_per_group': 16, 'norm_affine': True, 'reduction': 'sum', 'loss_weight': 1.0, 'learn_scale': True}
# ====================================
# GIoURegressor
# anchors_per_pos 27
# INFO Learning level specific scalar in regressor
# INFO Overwriting regressor conv weight init
# INFO Building:: head DetectionHeadHNMNative: {} sampler HardNegativeSamplerBatched: {'batch_size_per_image': 32, 'positive_fraction': 0.33, 'pool_size': 20, 'min_neg': 1}
# INFO Sampling hard negatives on a per batch basis
# INFO Building:: segmenter DiCESegmenterFgBg {'dice_kwargs': {'batch_dice': True}}
# INFO Running batch dice True and do bg False in dice loss.
# INFO Model Inference Summary: 
# detections_per_img: 100 
# score_thresh: 0 
# topk_candidates: 10000 
# remove_small_boxes: 0.01 
# nms_thresh: 0.6

# INFO Initialize SWA with swa epoch start 49
# INFO Using dummy 2d augmentation params
# INFO Running dummy 2d augmentation transforms!
# INFO Augmentation: BaseMoreAug transforms and base_more params 
# INFO Loading network patch size [ 16 224 224] and generator patch size [16, 320, 320]
# LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]