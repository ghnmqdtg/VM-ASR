#! /bin/bash

#Batch Job Paremeters
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=MST112230

python main.py \
    --cfg configs/vm_asr_16k_single_stream.yaml \
    --tag 16k_Full_Single_CG_smLR_Local

python main.py \
    --cfg configs/vm_asr_16k_single_stream_MPD.yaml \
    --tag 16k_Full_MPD_Single_CG_smLR_Local