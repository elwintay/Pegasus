#!/usr/bin/env python
# coding: utf-8
import transformers
import torch
import pandas as pd
import numpy as np  
import os
import re

text_path = 'Text Input/'
text_list = []
for text_file in os.listdir(text_path):
    
    new_path = text_path + text_file
    text_list.append(open(new_path, "r").read())

model_path = 'Pretrained/model/pegasus-original'
tokeniser_path = 'Pretrained/tokeniser/pegasus-tokeniser'

#If enough GPU RAM, use cuda, else just use cpu
#torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = transformers.PegasusTokenizer.from_pretrained(tokeniser_path)
model = transformers.PegasusForConditionalGeneration.from_pretrained(model_path)#.to(torch_device)

summary_list = []
for text in text_list:
    
    batch = tokenizer.prepare_seq2seq_batch(src_texts=[text])#.to(torch_device)
    gen = model.generate(**batch,max_length = 200, # max length of summary
                         min_length = 100, # min length of summary
                         do_sample = True, 
                         temperature = 3.0,
                         top_k =30,
                         top_p=0.70,
                         repetition_penalty = 1.2,
                         length_penalty = 3, # if more than 1 encourage model to generate #larger sequences
                         num_return_sequences=1 # no of summary you want to generate
                        )

    summary = tokenizer.batch_decode(gen, skip_special_tokens=True)
    print(summary[0])
    summary_list.append(summary[0])