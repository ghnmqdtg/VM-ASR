#! /bin/bash

#Batch Job Paremeters
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=MST112230

python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=1 \
    --master_addr="127.0.0.1" \
    --master_port=3427 main.py \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=1 \
    --cfg configs/vm_asr_basic.yaml