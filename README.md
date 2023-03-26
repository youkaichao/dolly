# Dolly

This fine-tunes the [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B) model on the [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset using a Databricks notebook.  Please note that while GPT-J 6B is [Apache 2.0 licensed](https://huggingface.co/EleutherAI/gpt-j-6B), the Alpaca dataset is licensed under [Creative Commons NonCommercial (CC BY-NC 4.0)](https://huggingface.co/datasets/tatsu-lab/alpaca).

## Get Started Training

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

* Add the `dolly` repo to Databricks (under Repos click Add Repo, enter `https://github.com/databrickslabs/dolly.git`, then click Create Repo).
* Start a `12.2 LTS ML (includes Apache Spark 3.3.2, GPU, Scala 2.12)` single-node cluster with node type having 8 A100 GPUs (e.g. `Standard_ND96asr_v4` or `p4d.24xlarge`).
* Open the `train_dolly` notebook in the `dolly` repo, attach to your GPU cluster, and run all cells.  When training finishes, the notebook will save the model under `/dbfs/dolly_training`.

## Running Unit Tests Locally

```
pyenv local 3.8.13
python -m venv .venv
. .venv/bin/activate
pip install -r requirements_dev.txt
./run_pytest.sh
```
