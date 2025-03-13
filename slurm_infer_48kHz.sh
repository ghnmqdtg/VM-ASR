#! /bin/bash

# # Test the versatile model for 48kHz output
# # Array of sample rates
# SAMPLE_RATES=(8000 12000 16000 24000)

# # Loop over sample rates and run the Python script
# for SR in "${SAMPLE_RATES[@]}"
# do
#     python inference.py \
#         --cfg configs/vm_asr_48k_MPD.yaml \
#         --resume logs/DualStreamInteractiveMambaUNet/48k_FullData_MPD \
#         --tag ${SR}_48000
# done

# Test the specialized model for 48kHz output
# Uncomment the following lines to test specialized models
# # 8kHz input, 48kHz output
# python inference.py \
#     --cfg configs/vm_asr_48k_MPD.yaml \
#     --resume logs/DualStreamInteractiveMambaUNet/48k_8k_FullData_MPD \
#     --input data/VCTK-Corpus-0.92/wav48_silence_trimmed_wav/p226/p226_004.wav \
#     --tag 8000_48000

# # 12kHz input, 48kHz output
# python inference.py \
#     --cfg configs/vm_asr_48k_MPD.yaml \
#     --resume logs/DualStreamInteractiveMambaUNet/48k_12k_FullData_MPD \
#     --input data/VCTK-Corpus-0.92/wav48_silence_trimmed_wav/p226/p226_004.wav \
#     --tag 12000_48000

# 16kHz input, 48kHz output
python main.py --cfg configs/vm_asr_48k_MPD.yaml \
    --resume logs/DualStreamInteractiveMambaUNet/48k_16k_FullData_MPD \
    --input data/VCTK-Corpus-0.92/wav48_silence_trimmed_wav/p226 \
    --inference \
    --tag 16000_48000

# # 24kHz input, 48kHz output
# python inference.py \
#     --cfg configs/vm_asr_48k_MPD.yaml \
#     --resume logs/DualStreamInteractiveMambaUNet/48k_24k_FullData_MPD \
#     --input data/VCTK-Corpus-0.92/wav48_silence_trimmed_wav/p226/p226_004.wav \
#     --tag 24000_48000