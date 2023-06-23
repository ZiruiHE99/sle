# SoftLE

Here, we provide the code of this paper: An Embarrassingly Simple Approach for Mitigating Undesired Shortcuts Using Soft Label Encoding.


# Getting Started

## Prerequisites:

python 3.8.10

pytorch 1.13.1

transformers 4.30.2

## Links for data used in this paper:

MNLI: https://huggingface.co/datasets/SetFit/mnli

HANS: https://huggingface.co/datasets/hans

FEVER: https://fever.ai/

FEVER-Symmetric: https://github.com/TalSchuster/FeverSymmetric

## How to run the code:

Here, we provide the pre-processed FEVER dataset and its corresponding challange datasets for convenience.

We have set the parameters according to the paper. You can run the following instructions to get the results in the paper. 

#train a teacher model(biased model):

python train_teacher.py

#train a student model(debiasing model):

python train_student.py

We conduct all the experiments on an A100 GPU. It takes about 2 hours either to train a teacher model or a student model. After the training of the teacher model, its weights will be saved, which will be utilized to train the student model. 

Here, we provide a set of weights: https://anonfiles.com/tdR0i0y4z6/bias_weights4
, then you can directly train the student model. 

