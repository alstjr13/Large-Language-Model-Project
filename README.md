**Note:** This repository is associated with MISR UBC Data Analytics & AI Research Group

### <ins> Packages, files needed to run the program:  </ins>

- Numpy
- Pandas
- transformers
- scikit-learn
- torch
- matplotlib
- seaborn
- utils.py
- cleanData.py

### <ins> Preliminary </ins>

3 different state-of-the-art language models were investigated, and fine-tuned to classify incentivized reviews from a given dataset.
- Bidirectional Encoder Representation from Transformers (BERT), Large, cased
- BigBird_RoBERTa_large
- Longformer_base_4096

The main focus of choosing these models were based on the max_length (input)

Bidirectional Encoder Representations from Transformers (BERT) was used to predict whether a review is incentivized or not incentivized using imported dataset from . 

Review dataset having approximately 11,000,000 data with review texts and labels were used to train the pre-trained models.

labels having 0 or 1: 0 indicating not incentivized and 1 meaning incentivized

### <ins> Terminology </ins>
**Incentivized Review**:

### <ins> Pre-processing Data </ins>

- reviews (including )
- sample reviews were gathered as .csv file from ~~~, and disclosure sentences were masked.

### <ins> Process of the Workflow: </ins>

- Raw data (in .csv file format)

### <ins> Hyperparameter, fine-tuning: </ins>

The best hyperparameters throughout the entire experiment were:

- learning_rate = 3e-5
  - Range tested: 1e-7 ~ 7e-5
- per_device_train_batch_size=32
  - Range tested: 8 ~ 64
- per_device_eval_batch_size=16
  - Range tested: 8 ~ 32
- adam_beta1=0.9
  - Range tested: 0.9 ~ 0.93
- adam_beta2=0.999
  - Range tested: 0.99 ~ 0.999


The focus of the experiment when altering hyperparameters were learning_rate, per_device_train_batch_size and per_device_eval_batch_size



Initially, hyperparameters were set to:

- 

### <ins> Further Studies: </ins>

To check whether the model is being trained efficiently and accurately, several things can be considered further:

- Loss gradient (optimization):
  - dd
  
- 
