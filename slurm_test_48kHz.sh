#! /bin/bash

#Batch Job Paremeters
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=MST112230

# Array of sample rates
# SAMPLE_RATES=(8000 12000 16000 24000)
SAMPLE_RATES=(16000)

# # Loop over sample rates and run the Python script
# for SR in "${SAMPLE_RATES[@]}"
# do
#     python main.py \
#         --cfg configs/vm_asr_48k_MPD.yaml \
#         --resume logs/DualStreamInteractiveMambaUNet/48k_FullData_MPD \
#         --eval \
#         --tag ${SR}_48000
# done

# Loop over sample rates and run the Python script
for SR in "${SAMPLE_RATES[@]}"
do
    python main.py \
        --cfg configs/vm_asr_48k_MPD_M2P.yaml \
        --resume logs/DualStreamInteractiveMambaUNet/48k_16k_FullData_MPD_M2P \
        --eval \
        --tag ${SR}_48000
done

# Loop over sample rates and run the Python script
for SR in "${SAMPLE_RATES[@]}"
do
    python main.py \
        --cfg configs/vm_asr_48k_MPD_P2M.yaml \
        --resume logs/DualStreamInteractiveMambaUNet/48k_16k_FullData_MPD_P2M \
        --eval \
        --tag ${SR}_48000
done