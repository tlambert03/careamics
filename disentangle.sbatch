#!/bin/bash
#SBATCH --job-name=careamics_training
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=federico.carrara@fht.org
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --output=/group/jug/federico/careamics_training/hpc_outputs/careamics_training_%j.log
#SBATCH --mem=128GB

# Check if mamba is installed
MAMBA_PATH=$(whereis mamba | awk '{print $2}')

# If mamba is not installed, MAMBA_PATH will be empty
if [ -z "$MAMBA_PATH" ]
then
    echo "Mamba is not installed. Running initialization..."
    ~/miniforge3/bin/conda init
    ~/miniforge3/bin/mamba init
else
    echo "Using mamba at: $MAMBA_PATH"
fi

source $HOME/.bashrc
mamba activate train_lvae_hpc
python /home/federico.carrara/Documents/projects/careamics/src/careamics/lvae_training/train_lvae.py --workdir=/group/jug/federico/careamics_training/training -mode=train --datadir=/group/jug/federico/careamics_training/data/BioSR --config=src/careamics/lvae_training/get_config.py 
sleep 5
wait
