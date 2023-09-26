import os
import json
import math
from collections import OrderedDict
import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import AdamW
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
import numpy as np
import gzip, csv
import pandas as pd
from tqdm.auto import tqdm
import wget

torch.manual_seed(0)
np.random.seed(0)


# %pip install transformers
from transformers import AutoTokenizer
# If you can not find all the bugs, use the line below for AutoModel
from transformers import AutoModel

 #TODO: add code to load NLI dataset in required format
 ###    if load_nli_dataset(..) is not appropriate for your method

class BertContrastive(nn.Module):
    #TODO: add __init__ to construct BertContrastive based on given pretrained BERT
    #TODO: add code for forward pass that returns the loss value
    #TODO: add aditional method if required



if __name__ == "__main__":
    #INFO: model and training configs
    model_name = 'prajjwal1/bert-tiny'
    num_epochs = 3
    train_batch_size = 8
    num_labels =3
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_config = {"hidden_size": 128, "num_attention_heads": 2, "num_hidden_layers": 2, "intermediate_size": 512, "vocab_size": 30522}
    bert_path = 'bert_tiny.bin'

    #WARNING: Change this code if you implemented a different nli loader for this part
    nli_dataset = load_nli_dataset('AllNLI.tsv.gz')

    #INFO: tokenize dataset
    #WARNING: Use only first 50000 samples and maximum sequence lenght of 128
    tokenized_train = tokenize_sentence_pair_dataset(nli_dataset['train'][:50000], tokenizer, max_length=128)

    #INFO: generate train_dataloader
    train_dataloader = get_dataloader(tokenized_train, batch_size=batch_size)

    #TODO: Create a BertContrastive with required parameters
    ###    Replace None with required input based on yor implementation
    bert_contrastive = BertContrastive(None)

    #INFO: create optimizer and run training loop
    optimizer = AdamW(bert_contrastive.parameters(), lr=5e-5)
    train_loop(bert_contrastive, optimizer, train_dataloader, num_epochs, device)
    
    #TODO: run evaluation loop
    result_from_contrastive = eval_loop(model, test_dataloader, device)
    print(f'\nPearson correlation: {result_from_contrastive[0]:.2f}\nSpearman correlation: {result_from_contrastive[1]:.2f}')