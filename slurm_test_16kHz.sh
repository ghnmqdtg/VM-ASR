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
        --cfg configs/vm_asr_16k_single_stream.yaml \
        --resume logs/VMUNet/16k_Full_Single_CG_smLR_Local \
        --eval \
        --tag ${SR}_16000
done

# Loop over sample rates and run the Python script
for SR in "${SAMPLE_RATES[@]}"
do
    python main.py \
        --cfg configs/vm_asr_16k_single_stream_MPD.yaml \
        --resume logs/VMUNet/16k_Full_MPD_Single_CG_smLR_Local \
        --eval \
        --tag ${SR}_16000
done