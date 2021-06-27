#!/usr/bin/env python
# coding: utf-8

import sys
import transformers
from transformers import Trainer, TrainingArguments
import torch
import pandas as pd
import numpy as np  
from Src import data_processing
from Src.prepare_dataset import Summary_dataset

text_path = 'Data/Text/'
summary_path = 'Data/Summary/'

full_data = data_processing.text_summary_to_csv(text_path,summary_path)
full_data = data_processing.text_processing(full_data)
train_texts,test_texts,train_decode,test_decode = data_processing.train_val_test_split(full_data,train_pct=0.8)

model_path = 'Pretrained/model/pegasus-original'
tokeniser_path = 'Pretrained/tokeniser/pegasus-tokeniser'

#If enough GPU RAM, use cuda, else just use cpu
#torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = transformers.PegasusTokenizer.from_pretrained(tokeniser_path)
model = transformers.PegasusForConditionalGeneration.from_pretrained(model_path)#.to(torch_device)

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_labels = tokenizer(train_decode, truncation=True, padding=True)
test_labels = tokenizer(test_decode, truncation=True, padding=True)

train_dataset = Summary_dataset(train_encodings, train_labels)
test_dataset = Summary_dataset(test_encodings, test_labels)


training_args = TrainingArguments(
    output_dir='./results',         # output directory
    num_train_epochs=5,              # total number of training epochs
    per_device_train_batch_size=64,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,
    logging_steps=1,               # strength of weight decay
    logging_dir='./logs',           # directory for storing logs
    overwrite_output_dir=True,
    no_cuda = True
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset             # evaluation dataset
)

trainer.train()
model.save_pretrained('Pretrained/model/pegasus-finetuned')