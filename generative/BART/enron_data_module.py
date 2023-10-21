import os 
import json 
import re 
import random 
from collections import defaultdict 
import argparse 

import transformers 
from transformers import BartTokenizer
import torch 
from torch.utils.data import DataLoader 
import pytorch_lightning as pl 
from sentence_transformers import SentenceTransformer, util

#import gensim.downloader as api
#import numpy as np
#import string

MAX_CONTEXT_LENGTH=1024 # measured in words 
MAX_LENGTH=1024
MAX_TGT_LENGTH=1024

from gen_data_loader import gen_with_templates_dataset, my_collate

# from pathlib import Path
# curr_file_path = Path(__file__).absolute()
# root_dir = str(curr_file_path.parent.parent.parent) + '/'

class enronDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__() 
        self.hparams = args
        #tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, truncation_side = "right", add_prefix_space=True, use_fast=False)
        #tokenizer.add_tokens([AddedToken(" <"), AddedToken('[TYPE]'), AddedToken("[CONTEXT]"), AddedToken('[EOT]'), AddedToken("[BOT]")])
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', truncation_side = "right", add_prefix_space=True)
        self.tokenizer.add_tokens(['[CONTEXT]', '[EOT]', '[BOT]'])
        self.tokenizer.end_of_template = '[EOT]'
        self.tokenizer.begin_of_template = '[BOT]'
    def train_dataloader(self):
        train_dataset = gen_with_templates_dataset('./data/train_', self.tokenizer, max_len=MAX_LENGTH)
        print('Training Dataset size: ', len(train_dataset))
        dataloader = DataLoader(train_dataset, pin_memory=True, num_workers=2, collate_fn=my_collate, batch_size=self.hparams.train_batch_size, shuffle = True)
        return dataloader
    def val_dataloader(self):
        val_dataset = gen_with_templates_dataset('./data/dev_', self.tokenizer, max_len=MAX_LENGTH)
        print('Val Dataset size: ', len(val_dataset))
        dataloader = DataLoader(val_dataset, pin_memory=True, num_workers=2, collate_fn=my_collate, batch_size=self.hparams.val_batch_size, shuffle = True)
        return dataloader
    def test_dataloader(self):
        test_dataset = gen_with_templates_dataset('./data/dev_', self.tokenizer, max_len=MAX_LENGTH)
        dataloader = DataLoader(test_dataset, pin_memory=True, num_workers=2, collate_fn=my_collate, batch_size=self.hparams.val_batch_size, shuffle = True)
        return dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--val_batch_size', type=int, default=4)
    args = parser.parse_args()
    dm = enronDataModule(args)
    dataloader = dm.train_dataloader()
    for idx, batch in enumerate(dataloader):
        print(batch)
        break 



