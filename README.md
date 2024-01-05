# LEGION

"*LEGION: Harnessing Pre-trained Language Models for GitHub Topic Recommendations with Distribution-Balance Loss*" (Submitted to EASE 2024)

by .

## Repository Organization
The structure of our source code's repository is as follows:
- data: contains our processeed data;
- scripts: contains scripts for RQ replication;
- src: contains our source code.
    - train.py: contains source code for fine-tuning
    - eval.py: contains source code for evaluating
    - util_loss.py: contains source code for utility functions
    - dataset_prep.py: contains source code for data preprocessing
    - ablation_eval.py: contains source code for our ablation study


## Installation

### Requirements
#### Hardware
- More than 50GB disk space
- GPU that has CUDA supports and have at least 16GB memory.
#### Software
- Ubuntu 18.04(?) or newer
- Conda

### Environment setup
- Python==3.7.6
  ```
  conda create -n LEGION python==3.7.6
  ```
- Packages
  ```
  pip3 install -r requirements.txt
  ```
- Pre-trained models
  ```
  mkdir berts
  cd berts
  git clone https://huggingface.co/bert-base-uncased
  git clone https://huggingface.co/roberta-base
  git clone https://huggingface.co/google/electra-base-discriminator
  git clone https://huggingface.co/facebook/bart-base 
  cd ..
  ```
- Others
  ```
  mkdir models logs
  ```


## General Usage

### Finetuning

For each pretrained model and loss function pair
```
python src/train.py {loss_function_name} {batch_size} {epochs} {pretrained_model_name}
# For example:
python src/train.py DBloss 16 40 roberta_base

```
The best model will be saved in the `models` folder and messages will be logged in the `logs` folder.

### Evaluation

With at least a suitable checkpoint in the 'models' folder:
```
python eval.py {pretrained_model_name} {1 to use threshold, 0 to not} {loss_function_names, separated by a blank space}
# For example:
python src/eval.py bert_base 0 BCE DBloss
```
This will generate a `.xlsx` file with the corresponding model name.


## RQ Replication
To best replicate the result of our experiments, you can download the finetuned models from our [replication package](), put them in the 'models' folder within this repository, then run only evaluation. The instructions below will finetune the pretrained models before evaluating them.

### RQ1
Each pre-trained model has its own script:
```
bash scripts/rq1_bert.sh
bash scripts/rq1_bart.sh
bash scripts/rq1_roberta.sh
bash scripts/rq1_electra.sh
```

### RQ2
To replicate results of the improved PTMs, run the scripts corresponding to each model: 
```
bash scripts/rq2_bert.sh
bash scripts/rq2_bart.sh
bash scripts/rq2_roberta.sh
bash scripts/rq2_electra.sh
```

### RQ3
```
bash scripts/rq3.sh
```

### RQ4
```
bash scripts/rq4.sh
```
