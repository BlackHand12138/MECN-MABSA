# MECN-MABSA

## Overview

This repository provides the implementation of MECN for multimodal aspect-based sentiment analysis (MABSA).

The code includes:

- the main training and evaluation script
- the MECN model implementation
- the RMT visual refinement module
- bidirectional multimodal cross-attention
- text-based and cross-modal graph reasoning
- adaptive gated fusion
- image feature extraction utilities
- data loading and preprocessing utilities

## File Structure

```text
.
├── train.py
├── MyModel.py
├── RMT.py
├── Gated_Fusion.py
├── img_deal_by_vit.py
├── data_utils.py
├── requirements.txt
├── README.md
├── LICENSE
└── DATA_NOTICE.md
```

## Environment

The experimental environment uses Python 3.10.19, PyTorch 2.9.1+cu128, Transformers 4.57.3, and CUDA 12.8.

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

## Data Preparation

Please prepare the datasets and related files locally before training. The code expects the following directory structure:

```text
data/
├── twitter2015/
│   ├── train.tsv
│   ├── dev.tsv
│   └── test.tsv
├── twitter2017/
│   ├── train.tsv
│   ├── dev.tsv
│   └── test.tsv
├── twitter2015_images/
├── twitter2017_images/
├── caption/
│   ├── twitter2015_images.json
│   └── twitter2017_images.json
├── face_descriptions/
│   ├── twitter2015_images_face.json
│   └── twitter2017_images_face.json
├── imgDealFile/
│   ├── twitter2015_images.pkl
│   └── twitter2017_images.pkl
├── oriAdj/
│   ├── train_ori_adj.pkl
│   ├── dev_ori_adj.pkl
│   └── test_ori_adj.pkl
└── HF/
    ├── config.json
    └── preprocessor_config.json
```

Cached data-loader files may be created or loaded from the following directory:

```text
middleFile/
├── twitter15_train_datas.pkl
├── twitter15_val_datas.pkl
├── twitter15_test_datas.pkl
├── twitter17_train_datas.pkl
├── twitter17_val_datas.pkl
└── twitter17_test_datas.pkl
```

The training script saves experimental outputs under `result/` and model checkpoints under `models/`.

## Running

Train and evaluate MECN on Twitter-2015:

```bash
python train.py --dataset twitter15 --model_name simpleBert --MAX_LEN 50 --BATCH_SIZE 32 --EPOCHS 20 --LEARNING_RATE 5e-5 --DEVICE cuda:0
```

Train and evaluate MECN on Twitter-2017:

```bash
python train.py --dataset twitter17 --model_name simpleBert --MAX_LEN 50 --BATCH_SIZE 32 --EPOCHS 20 --LEARNING_RATE 5e-5 --DEVICE cuda:0
```

## Notes

- The text encoder is based on `bert-base-uncased`.
- Visual features are extracted using a ViT-based encoder.
- The datasets, images, generated captions, facial descriptions, and intermediate feature files are not included in this repository.
- Please prepare the required resources according to the directory structure above.

## Reproducibility

This repository provides the model source code, training and evaluation script, dependency information, dataset path conventions, and preprocessing-related file descriptions.
