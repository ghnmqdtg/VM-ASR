#! /bin/bash

#Batch Job Paremeters
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=MST112230

# Array of sample rates
SAMPLE_RATES=(2000 4000 8000 12000)

# Loop over sample rates and run the Python script
for SR in "${SAMPLE_RATES[@]}"
do
    python main.py \
        --cfg configs/vm_asr_basic.yaml \
        --resume logs/DualStreamInteractiveMambaUNet/20240507030555 \
        --eval \
        --tag ${SR}_16000
done