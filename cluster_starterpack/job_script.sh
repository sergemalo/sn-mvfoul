#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4 
#SBATCH --mem=22000M                # REQUIRED: Otherwise the job is killed OOM
#SBATCH --account=def-sponsor00     # REQUIRED: Resource Allocation Project Identifier
#SBATCH --time=02:00:00             # Max time for the job
#SBATCH --job-name=testing_maxtime  # Job name to distinguish it with squeue

set -e

module load python # REQUIRED (e.g. to load av package), no need to load cuda (torch.cuda.is_available() returns True)

# Variables
SOURCEDIR=/home/edouardudem/sn-mvfoul
MODELDIR="$SOURCEDIR/VARS model"
REQUIREMENTS_FILE="$SOURCEDIR/cluster_starterpack/requirements-sm.txt"
DATASET_DIR="/project/def-sponsor00/francispicard2000"
DATASET_NAME=mvfouls-sub2-lr
WANDB_RUN_NAME=edouard_test_dataset

# Set up virtual environment and source directory
virtualenv "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"

# Pip the packages into the environment
pip install --ignore-installed -r "$REQUIREMENTS_FILE" # ignore_installed prevents trying to uninstall sympy/numpy on /cvmfs/soft.computecanada.ca/easybuild/...

# Transfer the data to SLURM_TMPDIR
mkdir "$SLURM_TMPDIR/data"
cp -r "$DATASET_DIR/$DATASET_NAME" "$SLURM_TMPDIR/data"

# Launch the code
export WANDB_API_KEY=$(grep -A2 'api.wandb.ai' ~/.netrc | grep password | awk '{print $2}') # Retrieve wandb key from the .netrc file
wandb login $WANDB_API_KEY
python "$MODELDIR/main.py" --max_epochs 10 --path "$SLURM_TMPDIR/data/$DATASET_NAME" --pre_model r3d_18 --data_aug No --batch_size 1 --wandb_run_name $WANDB_RUN_NAME
