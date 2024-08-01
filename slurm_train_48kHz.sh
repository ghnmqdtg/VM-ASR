#! /bin/bash

#Batch Job Paremeters
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=MST112230

# python main.py \
#     --cfg configs/vm_asr_48k_MPD.yaml \
#     --tag 48k_FullData_MPD

python main.py \
    --cfg configs/vm_asr_48k_MPD_M2P.yaml \
    --tag 48k_16k_FullData_MPD_M2P \
    --input_sr 16000

python main.py \
    --cfg configs/vm_asr_48k_MPD_P2M.yaml \
    --tag 48k_16k_FullData_MPD_P2M \
    --input_sr 16000