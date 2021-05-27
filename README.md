## Overview

This is the PyTorch implementation of paper [Multi-resolution CSI Feedback with deep learning in Massive MIMO System](https://arxiv.org/abs/1910.14322).

## Requirements

To use this project, you need to ensure the following requirements are installed.

- Python >= 3.7
- [PyTorch >= 1.2](https://pytorch.org/get-started/locally/)
- [thop](https://github.com/Lyken17/pytorch-OpCounter)

## Project Preparation

#### A. Data Preparation

The channel state information (CSI) matrix is generated from [COST2100](https://ieeexplore.ieee.org/document/6393523) model. Chao-Kai Wen and Shi Jin group provides a pre-processed version of COST2100 dataset in [Google Drive](https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj?usp=sharing), which is easier to use for the CSI feedback task; You can also download it from [Baidu Netdisk](https://pan.baidu.com/s/1Ggr6gnsXNwzD4ULbwqCmjA).

You can generate your own dataset according to the [open source library of COST2100](https://github.com/cost2100/cost2100) as well. The details of data pre-processing can be found in our paper.

#### B. Checkpoints Downloading

The model checkpoints should be downloaded if you would like to reproduce our result. All the checkpoints files can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1evKXkcF2Qp8Wn6cWJQiYQw) or [Google Drive](https://drive.google.com/drive/folders/16hQsrxkFuyjtmW4DOI8-Tix5TP5JfIia?usp=sharing)

#### C. Project Tree Arrangement

We recommend you to arrange the project tree as follows.

```
home
├── CRNet  # The cloned CRNet repository
│   ├── dataset
│   ├── models
│   ├── utils
│   ├── main.py
├── COST2100  # The data folder
│   ├── DATA_Htestin.mat
│   ├── ...
├── Experiments
│   ├── checkpoints  # The checkpoints folder
│   │     ├── in_04.pth
│   │     ├── ...
│   ├── run.sh  # The bash script
...
```

## Train CRNet from Scratch

An example of run.sh is listed below. Simply use it with `sh run.sh`. It will start advanced scheme aided CRNet training from scratch. Change scenario using `--scenario` and change compression ratio with `--cr`.

``` bash
python /home/CRNet/main.py \
  --data-dir '/home/COST2100' \
  --scenario 'in' \
  --epochs 2500 \
  --batch-size 200 \
  --workers 0 \
  --cr 4 \
  --scheduler cosine \
  --gpu 0 \
  2>&1 | tee log.out
```

## Results and Reproduction

The main results reported in our paper are presented as follows. All the listed results can be found in Table1 of our paper. They are achieved from training CRNet with our advanced training scheme (cosine annealing scheduler with warm up for 2500 epochs).


Scenario | Compression Ratio | NMSE | Flops | Checkpoints
:--: | :--: | :--: | :--: | :--:
indoor | 1/4 | -26.99 | 5.12M | in_04.pth
indoor | 1/8 | -16.01 | 4.07M | in_08.pth
indoor | 1/16 | -11.35 | 3.55M | in_16.pth
indoor | 1/32 | -8.93 | 3.28M | in_32.pth
indoor | 1/64 | -6.49 | 3.16M | in_64.pth
outdoor | 1/4 | -12.70 | 5.12M | out_04.pth
outdoor | 1/8 | -8.04 | 4.07M | out_08.pth
outdoor | 1/16 | -5.44 | 3.55M | out_16.pth
outdoor | 1/32 | -3.51 | 3.28M | out_32.pth
outdoor | 1/64 | -2.22 | 3.16M | out_64.pth

As aforementioned, we provide model checkpoints for all the results. Our code library supports easy inference. 

**To reproduce all these results, simple add `--evaluate` to `run.sh` and pick the corresponding pre-trained model with `--pretrained`.** An example is shown as follows.

``` bash
python /home/CRNet/main.py \
  --data-dir '/home/COST2100' \
  --scenario 'in' \
  --pretrained './checkpoints/in_04' \
  --evaluate \
  --batch-size 200 \
  --workers 0 \
  --cr 4 \
  --cpu \
  2>&1 | tee log.out
```

## Acknowledgment

Thank Chao-Kai Wen and Shi Jin group again for providing the pre-processed COST2100 dataset, you can find their related work named CsiNet in [Github-Python_CsiNet](https://github.com/sydney222/Python_CsiNet) 
