#! /bin/bash

#Batch Job Paremeters
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=MST113341

# Train the versatile model for 16kHz output
python main.py \
    --cfg configs/vm_asr_48k_MPD.yaml \
    --tag 48k_FullData_MPD

# Train the specialized model for 16kHz output
# Uncomment the following lines to train specialized models
# # 8kHz input, 48kHz output
# python main.py \
#     --cfg configs/vm_asr_48k_MPD.yaml \
#     --tag 48k_8k_FullData_MPD \
#     --input_sr 8000

# # 12kHz input, 48kHz output
# python main.py \
#     --cfg configs/vm_asr_48k_MPD.yaml \
#     --tag 48k_12k_FullData_MPD \
#     --input_sr 12000

# # 16kHz input, 48kHz output
# python main.py \
#     --cfg configs/vm_asr_48k_MPD.yaml \
#     --tag 48k_16k_FullData_MPD \
#     --input_sr 16000

# # 24kHz input, 48kHz output
# python main.py \
#     --cfg configs/vm_asr_48k_MPD.yaml \
#     --tag 48k_24k_FullData_MPD \
#     --input_sr 24000

# Ablation studies
# Uncomment the following lines to train ablation studies
# # GAN (X) & Post Processing (O)
# python main.py \
#     --cfg configs/vm_asr_48k.yaml \
#     --tag 48k_16k_FullData \
#     --input_sr 16000

# # GAN (O) & Post Processing (X)
# python main.py \
#     --cfg configs/vm_asr_48k_MPD_wo_POST.yaml \
#     --tag 48k_16k_FullData_MPD_wo_POST \
#     --input_sr 16000

# # GAN (X) & Post Processing (X)
# python main.py \
#     --cfg configs/vm_asr_48k_wo_POST.yaml \
#     --tag 48k_16k_FullData_wo_POST \
#     --input_sr 16000

# # Interactions: Magnitude to Phase
# python main.py \
#     --cfg configs/vm_asr_48k_MPD_M2P.yaml \
#     --tag 48k_16k_FullData_MPD_M2P \
#     --input_sr 16000

# # Interactions: Phase to Magnitude
# python main.py \
#     --cfg configs/vm_asr_48k_MPD_P2M.yaml \
#     --tag 48k_16k_FullData_MPD_P2M \
#     --input_sr 16000

# # Interactions: Magnitude alone
# python main.py \
#     --cfg configs/vm_asr_48k_MPD_SINGLE.yaml \
#     --tag 48k_16k_FullData_MPD_SINGLE \
#     --input_sr 16000

# # Hyperparams: N_FFT 2048
# python main.py \
#     --cfg configs/vm_asr_48k_16k_nfft2048.yaml \
#     --tag 48k_16k_FullData_MPD_NFFT2048 \
#     --input_sr 16000

# # Hyperparams: VSSM 8
# python main.py \
#     --cfg configs/vm_asr_48k_16k_MPD_VSSM8.yaml \
#     --tag 48k_16k_FullData_MPD_VSSM8 \
#     --input_sr 16000

# # Hyperparams: VSSM 24
# python main.py \
#     --cfg configs/vm_asr_48k_16k_MPD_VSSM24.yaml \
#     --tag 48k_16k_FullData_MPD_VSSM24 \
#     --input_sr 16000

# # Hyperparams: VSSM 32
# python main.py \
#     --cfg configs/vm_asr_48k_16k_MPD_VSSM32.yaml \
#     --tag 48k_16k_FullData_MPD_VSSM32 \
#     --input_sr 16000