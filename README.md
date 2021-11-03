## Notes
This is an implementation of time-frequency conditional discriminator

## Datasets

**Preparing Data**

- Download the training dataset. This can be any wav file with sampling rate 24,000Hz. The original paper used LibriTTS.
  - LibriTTS train-clean-360 split [tar.gz link](https://www.openslr.org/resources/60/train-clean-360.tar.gz)
  - Unzip and place its contents under `datasets/LibriTTS/train-clean-360`.
- If you want to use wav files with a different sampling rate, please edit the configuration file (see below).

Note: The mel-spectrograms calculated from audio file will be saved as `**.mel` at first, and then loaded from disk afterwards.

**Preparing Metadata**

Following the format from [NVIDIA/tacotron2](https://github.com/NVIDIA/tacotron2), the metadata should be formatted as:
```
path_to_wav|transcript|speaker_id
path_to_wav|transcript|speaker_id
...
```

Train/validation metadata for LibriTTS train-clean-360 split and are already prepared in `datasets/metadata`.
5% of the train-clean-360 utterances were randomly sampled for validation.

Since this model is a vocoder, the transcripts are **NOT** used during training.  

## Train

**Preparing Configuration Files**

- Run `cp config/default_c32.yaml config/config.yaml` and then edit `config.yaml`

- Write down the root path of train/validation in the `data` section. The data loader parses list of files within the path recursively.
  
  ```yaml
  data:
    train_dir: 'datasets/'	# root path of train data (either relative/absoulte path is ok)
    train_meta: 'metadata/libritts_train_clean_360_train.txt'	# relative path of metadata file from train_dir
    val_dir: 'datasets/'		# root path of validation data
    val_meta: 'metadata/libritts_train_clean_360_val.txt'		# relative path of metadata file from val_dir
  ```
  
  We provide the default metadata for LibriTTS train-clean-360 split.
  
- Modify `channel_size` in `gen` to switch between UnivNet-c16 and c32.
  
  ```yaml
  gen:
    noise_dim: 64
    channel_size: 32 # 32 or 16
    dilations: [1, 3, 9, 27]
    strides: [8, 8, 4]
    lReLU_slope: 0.2
  ```

**Training**

```bash
python trainer.py -c CONFIG_YAML_FILE -n NAME_OF_THE_RUN
```

**Tensorboard**

```bash
tensorboard --logdir logs/
```

If you are running tensorboard on a remote machine, you can open the tensorboard page by adding `--bind_all` option.

## Inference

```bash
python inference.py -p CHECKPOINT_PATH -i INPUT_MEL_PATH -o OUTPUT_WAV_PATH
```

## License

This code is licensed under BSD 3-Clause License.

We referred following codes and repositories.

- The overall structure of the repository is based on [https://github.com/seungwonpark/melgan](https://github.com/seungwonpark/melgan).
- [datasets/dataloader.py](./datasets/dataloader.py) from https://github.com/NVIDIA/waveglow (BSD 3-Clause License)
- [model/mpd.py](./model/mpd.py) from https://github.com/jik876/hifi-gan (MIT License)
- [model/lvcnet.py](./model/lvcnet.py) from https://github.com/zceng/LVCNet (Apache License 2.0)
- [utils/stft_loss.py](./utils/stft_loss.py) # Copyright 2019 Tomoki Hayashi #  MIT License (https://opensource.org/licenses/MIT)

## References

Papers

- *Jang et al.*, [UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminators for High-Fidelity Waveform Generation](https://arxiv.org/abs/2106.07889)
- *Zeng et al.*, [LVCNet: Efficient Condition-Dependent Modeling Network for Waveform Generation](https://arxiv.org/abs/2102.10815)
- *Kong et al.*, [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646)

Datasets

- [LibriTTS](https://openslr.org/60/)
