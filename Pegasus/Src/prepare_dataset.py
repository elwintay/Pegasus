#!/usr/bin/env python
# coding: utf-8

import transformers
from transformers import Trainer, TrainingArguments
import torch
import pandas as pd
import numpy as np  
import os
import re

class Summary_dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])  # torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings)
