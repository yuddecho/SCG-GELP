# SCG-GELP

## Introduction

SCG-GELP is an effective codon optimization strategy. 

## Quick Start

Colab can be borrowed to enforce SCG-GELP strategy. (https://colab.research.google.com/github/yuddecho/SCG-GELP/blob/main/SCG-GELP-Colab.ipynb)

## Usage (local)

### 下载项目

```bash
# https://docs.github.com/zh/repositories/working-with-files/managing-large-files/installing-git-large-file-storage
git clone git@github.com:yuddecho/SCG-GELP.git
```

### 设置环境

```bash
# nvidia-smi -> CUDA Version: 12.0
conda create --prefix /home/featurize/work/scg-gelp python=3.10 -y

conda activate /home/featurize/work/scg-gelp

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y

conda remove -p /home/featurize/work/scg-gelp --all
```



## Train



