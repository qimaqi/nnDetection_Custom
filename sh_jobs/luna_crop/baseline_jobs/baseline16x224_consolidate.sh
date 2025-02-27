#!/bin/bash
#SBATCH --job-name=16_luna_predict
#SBATCH --output=sbatch_log/baseline_nndet16x224_predict_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

#SBATCH --nodelist=bmicgpu07
#SBATCH --cpus-per-task=4
#SBATCH --mem 96GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liujiaying423@gmail.com

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

# Consolidatecreat a unified inference plan. 
#The following command will copy all the models and predictions from the folds. 
#By adding the sweep_ options, the empiricaly hyperparameter optimization across all folds can be started. 
#This will generate a unified plan for all models which will be used during inference.
#example: nndet_consolidate 000 RetinaUNetV001_D3V001_3d --sweep_boxes

# nndet_consolidate 017 RetinaUNetV001_D3V001_3d --sweep_boxes --shape=16_224_224 --num_folds=10

#For the final test set predictions simply select the best model according to the validation scores and run the prediction command below. 
#Data which is located in raw_splitted/imagesTs will be automatically preprocessed and predicted by running the following command:
# Example: nndet_predict 000 RetinaUNetV001_D3V001_3d --fold -1
#nndet_predict 017 RetinaUNetV001_D3V001_3d --fold -1 --shape=16_224_224 #no need?
#baseline16x224_consolidate

#nndet_eval 017 RetinaUNetV001_D3V001_3d -1 --boxes --analyze_boxes --shape=16_224_224

cd /usr/bmicnas02/data-biwi-01/lung_detection/nnDetection_Custom/projects/Task016_Luna/scripts
python prepare_eval_cpm.py RetinaUNetV001_D3V001_3d --shape=16_224_224