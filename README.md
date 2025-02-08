# VM-ASR
**VM-ASR: A Lightweight Dual-Stream U-Net Model for Efﬁcient Audio Super-Resolution**

- The official PyTorch implementation of VM-ASR, a model designed for high-fidelity audio super-resolution.
- This research has been published in the *IEEE/ACM Transactions on Audio, Speech, and Language Processing* on 24 January 2025. [Paper](https://ieeexplore.ieee.org/document/10852332), [Demo](https://ghnmqdtg.github.io/vm-asr-demo/).

## Abstract
Audio super-resolution (ASR), also known as bandwidth extension (BWE), aims to enhance the quality of low-resolution audio by recovering high-frequency components. However, existing methods often struggle to model harmonic relationships accurately and balance the inference speed and computational complexity. In this paper, we propose VM-ASR, a novel lightweight ASR model that leverages the Visual State Space (VSS) block to effectively capture global and local contextual information within audio spectrograms. This enables VM-ASR to model harmonic relationships more accurately, improving audio quality. Our experiments on the VCTK dataset demonstrate that VM-ASR consistently outperforms state-of-the-art methods in spectral reconstruction across various input-output sample rate pairs, achieving significantly lower Log-Spectral Distance (LSD) while maintaining a smaller model size (3.01M parameters) and lower computational complexity (2.98 GFLOPS). This makes VM-ASR not only a promising solution for real-time applications and resource-constrained environments but also opens up exciting possibilities in telecommunications, speech synthesis, and audio restoration.

<!-- Image -->
<p align="center">
    <img src="assets/Overview_v2.png" width="100%">
</p>

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Citation & Contact](#citation_contact)
- [Acknowledgements](#acknowledgements)

## Requirements
- Python 3.10+
- Conda 23.5.0
- CUDA 12.0 / 11.7 (optional)
- PyTorch 2.3.0 (CUDA 12.0) / 2.0.1 (CUDA 11.7)
- torchaudio 2.3.0 (CUDA 12.0) / 2.0.2 (CUDA 11.7)
- NVIDIA GPU

Note: When developing, the model is trained on a single NVIDIA GeForce RTX 4060 Ti GPU with 16GB memory locally. All the weights and experiment data are trained and evaluated on a single NVIDIA Tesla V100 GPU with 32GB memory provided by the National Center for High-performance Computing (NCHC) of National Applied Research Laboratories (NARLabs) in Taiwan. We did not implement the multi-GPU training in this repository.

## Installation
```shell
# 1. Clone the repository
$ git clone https://github.com/ghnmqdtg/VM-ASR

# 2. Navigate to the project directory
$ cd VM-ASR

# 3. Create conda env
$ conda create --name vm-asr python=3.10

# 4. Activate the env
$ conda activate vm-asr
    
# 5A. Install PyTorch 2.3.0 (CUDA 12.0)
$ conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# 5B. Install PyTorch 2.0.1 (CUDA 11.7)
$ conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

6. Install the package
$ pip install -r requirements.txt

# 7. Install the selective scan kernel (Source: https://github.com/MzeroMiko/VMamba)
$ cd kernels/selective_scan && pip install . && cd ../..
```

## Usage
The training and evaluation commands are provided in the script files. You can run the training and evaluation scripts using the following commands:

### Training
```shell
# 16kHz target sample rate
$ sh slurm_train_16kHz.sh

# 48kHz target sample rate
$ sh slurm_train_48kHz.sh
```

### Evaluation
```shell
# 16kHz target sample rate
$ sh slurm_test_16kHz.sh

# 48kHz target sample rate
$ sh slurm_test_48kHz.sh
```

### Pretrained Models
Please check the release page for the pretrained models. To use the pretrained models, you can download them and decompress them to the `./logs/DualStreamInteractiveMambaUNet` directory. For example, `logs/DualStreamInteractiveMambaUNet/16k_2k_FullData_MPD`. The pretrained models are trained on the VCTK dataset with the following configurations:

<table>
  <thead>
    <tr>
      <th>Target SR (kHz)</th>
      <th>Input SR (kHz)</th>
      <th>Model ID</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" colspan="4"><strong>Specialized Models</strong></td>
    </tr>
    <tr>
      <td align="center" rowspan="4">16</td>
      <td align="center">2</td>
      <td><code>16k_2k_FullData_MPD</code></td>
      <td>2 to 16kHz</td>
    </tr>
    <tr>
      <td align="center">4</td>
      <td><code>16k_4k_FullData_MPD</code></td>
      <td>4 to 16kHz</td>
    </tr>
    <tr>
      <td align="center">8</td>
      <td><code>16k_8k_FullData_MPD</code></td>
      <td>8 to 16kHz</td>
    </tr>
    <tr>
      <td align="center">12</td>
      <td><code>16k_12k_FullData_MPD</code></td>
      <td>12 to 16kHz</td>
    </tr>
    <tr>
      <td align="center" rowspan="4">48</td>
      <td align="center">8</td>
      <td><code>48k_8k_FullData_MPD</code></td>
      <td>8 to 48kHz</td>
    </tr>
    <tr>
      <td align="center">12</td>
      <td><code>48k_12k_FullData_MPD</code></td>
      <td>12 to 48kHz</td>
    </tr>
    <tr>
      <td align="center">16</td>
      <td><code>48k_16k_FullData_MPD</code></td>
      <td>16 to 48kHz</td>
    </tr>
    <tr>
      <td align="center">24</td>
      <td><code>48k_24k_FullData_MPD</code></td>
      <td>24 to 48kHz</td>
    </tr>
    <tr>
      <td align="center" colspan="4"><strong>Versatile Models</strong></td>
    </tr>
    <tr>
      <td align="center">16</td>
      <td align="center">2~16</td>
      <td><code>16k_FullData_MPD</code></td>
      <td>2~16 to 16kHz</td>
    </tr>
    <tr>
      <td align="center">48</td>
      <td align="center">8~48</td>
      <td><code>48k_FullData_MPD</code></td>
      <td>8~48 to 48kHz</td>
    </tr>
    <tr>
      <td align="center" colspan="4"><strong>Ablation Study</strong></td>
    </tr>
    <tr>
      <td align="center" rowspan="5">48</td>
      <td align="center" rowspan="5">16</td>
      <td><code>48k_16k_FullData</code></td>
      <td>GAN (X) & Post Processing (O)</td>
    </tr>
    <tr>
      <td><code>48k_16k_FullData_MPD_woPost</code></td>
      <td>GAN (O) & Post Processing (X)</td>
    </tr>
    <tr>
      <td><code>48k_16k_FullData_woPost</code></td>
      <td>GAN (X) & Post Processing (X)</td>
    </tr>
    <tr>
      <td><code>48k_16k_FullData_MPD_M2P</code></td>
      <td>Interactions: Magnitude to Phase</td>
    </tr>
    <tr>
      <td><code>48k_16k_FullData_MPD_P2M</code></td>
      <td>Interactions: Phase to Magnitude</td>
    </tr>
  </tbody>
</table>

### Q&A
1. **Can I run the training commands directly?**

    Yes, of course. You can run the training commands directly without using the script files like this:

    ```shell
    Usage: python main.py [OPTIONS]

    Options:
        --cfg FILE                   path to config file [required]
        --batch-size INT             batch size for single GPU
        --input_sr INT               the input sample rate (if set, the random resample will be disabled)
        --resume PATH                path to checkpoint for models to resume training or run evaluation
        --disable_amp                Disable pytorch automatic mixed precision
        --output PATH                root of output folder, the full path is <output>/<model_name>/<tag>
        --tag TAG                    tag of experiment [default: current timestamp]
        --eval                       Perform evaluation only
    ```

2. **Why do I get an error message like `: not found_16kHz.sh: 2:`?**

    Please note that **the end-of-line sequence in the script files should be LF** instead of CRLF. If you encounter this error message, please check the line endings of the script files first.

2. **What's the model training by default?**

    The script files used to train and evaluate the versatile models (VM-ASR*) by default. If you want to train the specialized models (VM-ASR), you can uncomment the corresponding lines in the script files. The description of commands is provided in the script files.

3. **Waaait a minute, where is the dataset?**

    After you run the training script, the system will check if the dataset exists. If it is not found, the system will download, decompress, and convert it to WAV files **automatically**. If it is found, the system will skip such steps. Please note that the dataset is large (about 35GB, zip, FLAC, and WAV included), so the process may take some time.

## Configuration
### Configuration Files
The default configuration is provided in the `./config.py` file. We use this default configuration as the base configuration, and **we override it with the configuration file in `.yaml` format for each experiment.** These yaml files are located at `./config`.

Some frequently used hyperparameter you may want to adjust in yaml files:

1. `DATA.BATCH_SIZE`

    The batch size in configs are set for 32GB memory GPU. If you have a different GPU with smaller memory, you may need to adjust the batch size accordingly.

2. `DATA.USE_QUANTITY`

    This controls how much data you want to use for training and evaluation. The default value is 1.0, which means using all the data. You can adjust this value to a smaller number like 0.1 for debugging new features.

3. `TRAIN.ADVERSARIAL.ENABLE`

    This controls whether to enable the adversarial training. The default value is `False`. You can set it to `True` to enable the adversarial training.

### Logging

1. **Tensorboard**
    
    Tensorboard logging is enabled. The logs are saved in the `./logs` directory, and you can visualize them using the following command:

    ```shell
    # Launch Tensorboard
    $ tensorboard --logdir logs/
    ```

2. **Weights and Biases (WandB)**

    [Weights and Biases (Wandb)](https://wandb.ai/site) logging is used for syncing the training process online. It's enabled by default. You need to login to your WandB account for the first time running the training script. You can disable it in script files by setting `WANDB.ENABLE` to `False` if you don't want to use it.


## Citation & Contact
### Citation
```
@ARTICLE{10852332,
  author={Zhang, Ting-Wei and Ruan, Shanq-Jang},
  journal={IEEE Transactions on Audio, Speech and Language Processing}, 
  title={VM-ASR: A Lightweight Dual-Stream U-Net Model for Efficient Audio Super-Resolution}, 
  year={2025},
  volume={33},
  number={},
  pages={666-677},
  doi={10.1109/TASLPRO.2025.3533365}
}
```

### Contact
If you find any bugs or have any suggestions, please feel free to open an issue or submit a pull request. You can also email us at ghmmqdtg@gmail.com.

## Acknowledgements
We thank to National Center for High-performance Computing (NCHC) of National Applied Research Laboratories (NARLabs) in Taiwan for providing computational and storage resources.
