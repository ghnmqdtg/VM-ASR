#! /bin/bash

#Batch Job Paremeters
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=MST112230

# Test the versatile model for 16kHz output
# Array of sample rates
SAMPLE_RATES=(2000 4000 8000 12000)

# Loop over sample rates and run the Python script
for SR in "${SAMPLE_RATES[@]}"
do
    python main.py \
        --cfg configs/vm_asr_16k.yaml \
        --resume logs/DualStreamInteractiveMambaUNet/16k_FullData_MPD \
        --eval \
        --tag ${SR}_16000
done

# Test the specialized model for 16kHz output
# Uncomment the following lines to test specialized models
# # 2kHz input, 16kHz output
# python main.py \
#     --cfg configs/vm_asr_16k.yaml \
#     --resume logs/DualStreamInteractiveMambaUNet/16k_2k_FullData_MPD \
#     --eval \
#     --tag 2000_16000

# # 4kHz input, 16kHz output
# python main.py \
#     --cfg configs/vm_asr_16k.yaml \
#     --resume logs/DualStreamInteractiveMambaUNet/16k_4k_FullData_MPD \
#     --eval \
#     --tag 4000_16000

# # 8kHz input, 16kHz output
# python main.py \
#     --cfg configs/vm_asr_16k.yaml \
#     --resume logs/DualStreamInteractiveMambaUNet/16k_8k_FullData_MPD \
#     --eval \
#     --tag 8000_16000

# # 12kHz input, 16kHz output
# python main.py \
#     --cfg configs/vm_asr_16k.yaml \
#     --resume logs/DualStreamInteractiveMambaUNet/16k_12k_FullData_MPD \
#     --eval \
#     --tag 12000_16000