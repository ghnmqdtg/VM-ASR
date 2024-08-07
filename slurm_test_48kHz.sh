#! /bin/bash

#Batch Job Paremeters
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=MST112230

# Test the versatile model for 48kHz output
# Array of sample rates
SAMPLE_RATES=(8000 12000 16000 24000)

# Loop over sample rates and run the Python script
for SR in "${SAMPLE_RATES[@]}"
do
    python main.py \
        --cfg configs/vm_asr_48k_MPD.yaml \
        --resume logs/DualStreamInteractiveMambaUNet/48k_FullData_MPD \
        --eval \
        --tag ${SR}_48000
done

# Test the specialized model for 48kHz output
# Uncomment the following lines to test specialized models
# # 8kHz input, 48kHz output
# python main.py \
#     --cfg configs/vm_asr_48k_MPD.yaml \
#     --resume logs/DualStreamInteractiveMambaUNet/48k_8k_FullData_MPD \
#     --eval \
#     --tag 8000_48000

# # 12kHz input, 48kHz output
# python main.py \
#     --cfg configs/vm_asr_48k_MPD.yaml \
#     --resume logs/DualStreamInteractiveMambaUNet/48k_12k_FullData_MPD \
#     --eval \
#     --tag 12000_48000

# # 16kHz input, 48kHz output
# python main.py \
#     --cfg configs/vm_asr_48k_MPD.yaml \
#     --resume logs/DualStreamInteractiveMambaUNet/48k_16k_FullData_MPD \
#     --eval \
#     --tag 16000_48000

# # 24kHz input, 48kHz output
# python main.py \
#     --cfg configs/vm_asr_48k_MPD.yaml \
#     --resume logs/DualStreamInteractiveMambaUNet/48k_24k_FullData_MPD \
#     --eval \
#     --tag 24000_48000

# Ablation studies
# Uncomment the following lines to test ablation studies
# # GAN (X) & Post Processing (O)
# python main.py \
#     --cfg configs/vm_asr_48k.yaml \
#     --resume logs/DualStreamInteractiveMambaUNet/48k_16k_FullData \
#     --eval \
#     --tag 16000_48000

# # GAN (O) & Post Processing (X)
# python main.py \
#     --cfg configs/vm_asr_48k_MPD_wo_POST.yaml \
#     --resume logs/DualStreamInteractiveMambaUNet/48k_16k_FullData_MPD_wo_POST \
#     --eval \
#     --tag 16000_48000

# # GAN (X) & Post Processing (X)
# python main.py \
#     --cfg configs/vm_asr_48k_wo_POST.yaml \
#     --resume logs/DualStreamInteractiveMambaUNet/48k_16k_FullData_wo_POST \
#     --eval \
#     --tag 16000_48000

# # Interactions: Magnitude to Phase
# python main.py \
#     --cfg configs/vm_asr_48k_MPD_M2P.yaml \
#     --resume logs/DualStreamInteractiveMambaUNet/48k_16k_FullData_MPD_M2P \
#     --eval \
#     --tag 16000_48000

# # Interactions: Phase to Magnitude
# python main.py \
#     --cfg configs/vm_asr_48k_MPD_P2M.yaml \
#     --resume logs/DualStreamInteractiveMambaUNet/48k_16k_FullData_MPD_P2M \
#     --eval \
#     --tag 16000_48000
