#! /bin/bash

#Batch Job Paremeters
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=MST112230

# python main.py \
#     --cfg configs/vm_asr_16k_single_stream_MPD.yaml \
#     --tag 16k_Deci_single_stream_MPD_smlr

# python main.py \
#     --cfg configs/vm_asr_16k_single_stream_MPD.yaml \
#     --tag 16k_Deci_single_stream_MPD

# python main.py \
#     --cfg configs/vm_asr_16k_single_stream.yaml \
#     --tag 16k_Deci_single_stream

# python main.py \
#     --cfg configs/vm_asr_16k_single_stream_deep_MPD.yaml \
#     --tag 16k_Deci_single_stream_deep_MPD

# python main.py \
#     --cfg configs/vm_asr_16k_single_stream_deep.yaml \
#     --tag 16k_Deci_single_stream_deep

# python main.py \
#     --cfg configs/vm_asr_16k_MPD.yaml \
#     --tag 16k_FullData_MPD_Local

# python main.py \
#     --cfg configs/vm_asr_16k_MPD_CG.yaml \
#     --tag 16k_Deci_MPD_CG_Local

# python main.py \
#     --cfg configs/vm_asr_16k_L_MPD.yaml \
#     --tag 16k_Deci_L_MPD_CG_Local

python main.py \
    --cfg configs/vm_asr_16k_MPD_sox.yaml \
    --tag 16k_Deci_MPD_sox_Local

python main.py \
    --cfg configs/vm_asr_16k_MPD_GP.yaml \
    --tag 16k_Deci_MPD_GP_Local

python main.py \
    --cfg configs/vm_asr_16k_MPD_CG_P2M.yaml \
    --tag 16k_Deci_MPD_CG_P2M_Local

# python main.py \
#     --cfg configs/vm_asr_16k.yaml \
#     --tag 16k_Deci

# python main.py \
#     --cfg configs/vm_asr_16k_S_MPD.yaml \
#     --tag 16k_Deci_S_MPD_Delay

# # Train with individual input sample rates
# SAMPLE_RATES_IN_K=(2 4 8 12)

# # Loop over sample rates and run the Python script
# for SR_IN_K in "${SAMPLE_RATES_IN_K[@]}"
# do
#     python main.py \
#         --cfg configs/vm_asr_16k.yaml \
#         --tag 16k_${SR_IN_K}k_Deci_MPD
# done
