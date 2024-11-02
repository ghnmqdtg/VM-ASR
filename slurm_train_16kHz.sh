#! /bin/bash

#Batch Job Paremeters
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=MST113341

# Train the versatile model for 16kHz output
python main.py \
    --cfg configs/vm_asr_16k.yaml \
    --tag 16k_FullData_MPD

# Train the specialized model for 16kHz output
# Uncomment the following lines to train specialized models
# # 2kHz input, 16kHz output
# python main.py \
#     --cfg configs/vm_asr_16k.yaml \
#     --tag 16k_2k_FullData_MPD \
#     --input_sr 2000

# # 4kHz input, 16kHz output
# python main.py \
#     --cfg configs/vm_asr_16k.yaml \
#     --tag 16k_4k_FullData_MPD \
#     --input_sr 4000

# # 8kHz input, 16kHz output
# python main.py \
#     --cfg configs/vm_asr_16k.yaml \
#     --tag 16k_8k_FullData_MPD \
#     --input_sr 8000

# # 12kHz input, 16kHz output
# python main.py \
#     --cfg configs/vm_asr_16k.yaml \
#     --tag 16k_12k_FullData_MPD \
#     --input_sr 12000
