# Humor Recognition and Humor Response Generation in Dialogue Scenarios

## Introduction
* Partial code implementation for the First Prize winner of [CCL2022-The 4th "NiuTrans Cup" Humor Computing-Humor Sense of Chatbot Challenge](http://cips-cl.org/static/CCL2022/cclEval/humorcomputation/index.html)


## Requirements：
* python 3.6.8
* pytorch 1.6.0
* transformers 4.18.0
* CUDA version 10.1


## Humor Recognition
We explore sentence-based text classification, dialogue modeling, and prompt-based text classification. We conduct experiments using various BERT variants and apply a range of tricks. Then, we ensemble heterogeneous models through a stacking approach to further improve the performance.

* Bart_humor: encoding sentences with BART and modeling dialogue with a transformer encoder

* BertBaseline: a simple baseline using BERT to classify sentences

* BertPrompt: use prompt template to perform classification

* Bert_dialogue_modeling: encoding sentences with BERT and modeling dialogue with a transformer encoder

* Bert_utter_fgm_hyperopt: introduce context and apply tricks like focal loss, label_smoothing and fgm; search hyperparameters using wandb

* Continue_pretrain: crawl domain data from internet to continue pretrain BERT

* DataPreprocess: preprocess data to introduce the speaker information

* Data_augmentation: data augmentation using back translation

* TextCNN: classify with textCNN

* Voting: simple voting to ensemble heterogeneous models



## Humor Response Generation
We design a sliding window strategy to perform data augmentation and conduct experiments on GPT2 and T5. Furthermore, we utilize the humor recognition model to select humorous response from sampled results.

*  data/get_dailydialog.py: get dailydialog dataset
*  data/dataset.py: augment training data with a sliding window strategy
*  data/organize_humor_data.py: preprocess data for inference
*  humor_data: train dataset and dev dataset
*  test_result: save predict result

```
python T5-finetune.py    #start training
python predict.py        #start predict
```