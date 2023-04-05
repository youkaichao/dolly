# Dolly

This fine-tunes the [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B) model on the [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset using a Databricks notebook.  Please note that while GPT-J 6B is [Apache 2.0 licensed](https://huggingface.co/EleutherAI/gpt-j-6B), the Alpaca dataset is licensed under [Creative Commons NonCommercial (CC BY-NC 4.0)](https://huggingface.co/datasets/tatsu-lab/alpaca).

## Get Started Training

### Install dependency

```
pip install -r requirements_dev.txt
```

### Download dataset file and model file

Download from Tsinghua Cloud:
```
wget https://github.com/youkaichao/dolly/archive/refs/heads/master.zip
unzip master.zip
cd dolly-master

wget "https://cloud.tsinghua.edu.cn/f/498512c3c1724558830d/?dl=1" -O parquet-train.arrow
mkdir -p ./model
pushd ./model
wget "https://cloud.tsinghua.edu.cn/f/8bfd19e6cb1a4a289c1b/?dl=1" -O added_tokens.json
wget "https://cloud.tsinghua.edu.cn/f/231ddebf6caf49b38ce8/?dl=1" -O config.json
wget "https://cloud.tsinghua.edu.cn/f/79e402dcc503430db9a1/?dl=1" -O merges.txt
wget "https://cloud.tsinghua.edu.cn/f/001e6641d7324635bc77/?dl=1" -O special_tokens_map.json
wget "https://cloud.tsinghua.edu.cn/f/2d68e62358da4b7f94e6/?dl=1" -O tokenizer.json
wget "https://cloud.tsinghua.edu.cn/f/5ebcd4f2380147e3bee8/?dl=1" -O tokenizer_config.json
wget "https://cloud.tsinghua.edu.cn/f/34aa75355590497ba28b/?dl=1" -O vocab.json
wget "https://cloud.tsinghua.edu.cn/f/cd59c04366674ab592b0/?dl=1" -O pytorch_model.bin
popd
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

```python
model_path = '/path/to/checkpoint'
from training.generate import load_model_tokenizer_for_generate, generate_response
model, tokenizer = load_model_tokenizer_for_generate(model_path)
instruction='Write a tweet to introduce Dolly, a model to mimic ChatGPT.'
response = generate_response(instruction, model, tokenizer)
print(response)
```

(It is recommended to use `ipython` to interactively generate sentences to avoid loading models from disk again and again.)
