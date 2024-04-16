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
# python train.py --config config/config_20240408_v2_chunk_MAE_fullband.json
# python train.py --config config/config_20240409_v2_chunk_MAE_fullband_fulldata.json
# python train.py --config config/config_20240410_v2_chunk_MSE_large.json
# python train.py --config config/config_20240410_v2_chunk_MAE_ouputv1.json
# python train.py --config config/config_20240410_v2_chunk_MAE_ouputv2.json
# python train.py --config config/config_20240411_cat.json
# python train.py --config config/config_20240412_cat_outputv3.json

# The following configs aren't compatible to older versions of the code
# python train.py --config config/config_20240413_cat_outputv1_log.json
# python train.py --config config/config_20240413_cat_outputv1_dB.json
# python train.py --config config/config_20240413_cat_outputv2_log.json
# python train.py --config config/config_20240413_cat_outputv2_dB.json
# python train.py --config config/config_20240413_cat_outputv3_log.json
# python train.py --config config/config_20240413_cat_outputv3_dB.json
# python train.py --config config/config_20240413_cat_outputv3_log_randomlpf.json
python train.py --config config/config_20240413_add_outputv3_log.json
python train.py --config config/config_20240413_add_outputv3_log_randomlpf.json


