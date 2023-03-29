# Dolly

This fine-tunes the [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B) model on the [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset using a Databricks notebook.  Please note that while GPT-J 6B is [Apache 2.0 licensed](https://huggingface.co/EleutherAI/gpt-j-6B), the Alpaca dataset is licensed under [Creative Commons NonCommercial (CC BY-NC 4.0)](https://huggingface.co/datasets/tatsu-lab/alpaca).

## Get Started Training

### Install dependency

```
pip install -r requirements_dev.txt
```

### Download dataset file and model file

Download from Tsinghua Cloud: https://cloud.tsinghua.edu.cn/d/0185c787cdc243d1a3b7/ .

Alternatively, you can download model checkpoint from huggingface

```
pip install transformers
transformers-cli download EleutherAI/gpt-j-6B --cache-dir ./model/
export TRANSFORMERS_CACHE=`pwd`/model
```

### Train the model

* Start a single-node cluster with node type having 8 A100 (40GB memory) GPUs (e.g. `Standard_ND96asr_v4` or `p4d.24xlarge`).

```bash
export timestamp=`date +%Y-%m-%d_%H-%M-%S`
export model_name='dolly'
export checkpoint_dir_name="${model_name}__${timestamp}"
export deepspeed_config=`pwd`/config/ds_z3_bf16_config.json
export local_training_root='./'
export local_output_dir="${local_training_root}/${checkpoint_dir_name}"
export dbfs_output_dir=''
export tensorboard_display_dir="${local_output_dir}/runs"
export DATASET_FILE_PATH=`pwd`/parquet-train.arrow
export MODEL_PATH=`pwd`/model/
deepspeed --num_gpus=8 \
    --module training.trainer \
    --deepspeed $deepspeed_config \
    --epochs 1 \
    --local-output-dir $local_output_dir \
    --dbfs-output-dir "" \
    --per-device-train-batch-size 8 \
    --per-device-eval-batch-size 8 \
    --lr 1e-5
```

## Generate some sentences

```
python generate.py
```

(It is recommended to use `ipython` to interactively generate sentences to avoid loading models from disk again and again.)
