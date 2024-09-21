**Note:** This repository is associated with MISR UBC Data Analytics & AI Research Group

## Preliminary

3 different state-of-the-art language models were investigated, and fine-tuned to classify incentivized reviews from a given dataset.
- Bidirectional Encoder Representation from Transformers (BERT), Large, cased
- BigBird_RoBERTa_large
- Longformer_base_4096

The main focus of choosing these models were based on the max_length (input)

## Terminology
**Incentivized Review**: 

Bidirectional Encoder Representations from Transformers (BERT) was used to predict whether a review is incentivized or not incentivized using imported dataset form Qiao et al. (2020).

## Pre-processing Data
- reviews (including )
- sample reviews were gathered as .csv file from ~~~, and disclosure sentences were masked.
<<<<<<< HEAD


## BERT Specification
- hidden_size: 1024,
- intermediate_size: 4096
- num_attention_heads: 16
- num_hidden_layers: 24
- hidden_act: "gelu"

## Process of the Workflow
- Raw data (in .csv file format) 

## Hyperparameter, fine-tuning

- learning_rate = 3e-5
- per_device_train_batch_size=32
- per_device_eval_batch_size=16
- adam_beta1=0.9
- adam_beta2=0.999

The focus of the experiment when altering hyperparameters were learning_rate, per_device_train_batch_size and per_device_eval_batch_size



Initially, hyperparameters were set to:
-

