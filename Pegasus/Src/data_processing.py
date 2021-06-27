#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np  
import os
import re

def text_summary_to_csv(text_path,summary_path):
    
    text_data = pd.DataFrame()
    summary_data = pd.DataFrame()
    text_idx = []
    summary_idx = []
    texts = []
    summaries = []


    for text_file in os.listdir(text_path):
        new_path = text_path + text_file
        text_idx.append(text_file)
        texts.append(open(new_path, "r").read())

    text_data['id'] = text_idx
    text_data['Text'] = texts

    for summary_file in os.listdir(summary_path):
        new_path = summary_path + summary_file
        summary_idx.append(summary_file)
        summaries.append(open(new_path, "r").read())

    summary_data['id'] = summary_idx
    summary_data['Text'] = summaries
    
    full_data = text_data.merge(summary_data, on='id', how='left')
    full_data.columns = ['id','Text','Summary']
    
    return full_data


def text_processing(data):
    data['Text'] = data['Text'].apply(lambda x: re.sub('\n\n','. ', x))
    data['Summary'] = data['Summary'].apply(lambda x: re.sub('\n\n','. ', x))
    # data['Text'] = data['Text'].apply(lambda x: x.lower())
    # data['Summary'] = data['Summary'].apply(lambda x: x.lower())
    return data.reset_index(drop=True)


def train_val_test_split(in_df,train_pct):

    in_df = in_df.sample(len(in_df), random_state=20)
    train_sub = int(len(in_df) * train_pct)

    train_df = in_df[0:train_sub]
    test_df = in_df[train_sub:]

    train_texts = list(train_df['Text'])
    test_texts = list(test_df['Text'])

    train_decode = list(train_df['Summary'])
    test_decode = list(test_df['Summary'])
    
    return train_texts,test_texts,train_decode,test_decode
