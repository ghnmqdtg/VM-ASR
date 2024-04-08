#! /bin/bash

#Batch Job Paremeters
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=MST112230

# python train.py --config config/config_20240330_v2_chunk_StepLR.json
# python train.py --config config/config_20240330_v2_chunk_CosineWarmUp.json
# python train.py --config config/config_20240330_v2_full_StepLR.json
# python train.py --config config/config_20240330_v2_full_CosineWarmUp.json
# python train.py --config config/config_20240330_v1_full_StepLR_GAN.json
# python train.py --config config/config_20240330_v1_full_CosineWarmUp_GAN.json
# python train.py --config config/config_20240405_v2_chunk_MAE.json
# python train.py --config config/config_20240405_v2_chunk_MSE.json
# python train.py --config config/config_20240407_v2_chunk_MSE_fullband.json
python train.py --config config/config_20240407_v2_chunk_MAE_fullband.json
