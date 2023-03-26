# Dolly

This fine-tunes the [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B) model on the [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset using a Databricks notebook.  Please note that while GPT-J 6B is [Apache 2.0 licensed](https://huggingface.co/EleutherAI/gpt-j-6B), the Alpaca dataset is licensed under [Creative Commons NonCommercial (CC BY-NC 4.0)](https://huggingface.co/datasets/tatsu-lab/alpaca).

## Get Started Training

### Install dependency

```
pip install -r requirements_dev.txt
```

### Download dataset file and model file

```
wget https://cloud.tsinghua.edu.cn/seafhttp/files/beac2337-e002-403d-befa-4666db860078/parquet-train.arrow
mkdir ./model/ && cd ./model/
wget https://cloud.tsinghua.edu.cn/seafhttp/files/56f50dc0-4cbb-473d-ae95-6f9f87564b3c/vocab.json
wget https://cloud.tsinghua.edu.cn/seafhttp/files/22be3e9c-313d-4194-b513-dfda505aa7a0/tokenizer_config.json
wget https://cloud.tsinghua.edu.cn/seafhttp/files/83ce4dd2-7b00-42ec-b958-369f01573003/tokenizer.json
wget https://cloud.tsinghua.edu.cn/seafhttp/files/eb3f574f-a83e-45a5-b88e-f285d48f266f/special_tokens_map.json
wget https://cloud.tsinghua.edu.cn/seafhttp/files/69a4874c-e8b8-43f5-81a5-84cedc7a6f84/merges.txt
wget https://cloud.tsinghua.edu.cn/seafhttp/files/dc70c876-95f0-48c7-8082-26ca8f39ce06/config.json
wget https://cloud.tsinghua.edu.cn/seafhttp/files/4f79b941-5569-49b0-9b7b-6f6140788689/added_tokens.json
wget https://cloud.tsinghua.edu.cn/seafhttp/files/e5980d8d-1457-424f-a91d-906b8ed45aeb/pytorch_model.bin
```

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
export deepspeed_config = `pwd`/config/ds_z3_bf16_config.json
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
