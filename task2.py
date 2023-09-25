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

torch.manual_seed(0)
np.random.seed(0)


# %pip install transformers
from transformers import AutoTokenizer
# If you can not find all the bugs, use the line below for AutoModel
from transformers import AutoModel



if __name__ == "__main__":
    data = pd.read_csv('stsbenchmark.tsv.gz', nrows=5, compression='gzip', delimiter='\t')
    data.head()

    def load_sts_dataset(file_name):
        #TODO: add code to load STS dataset in required format
        sts_samples = {'test': []}
        return sts_samples


    def tokenize_sentence_pair_dataset(dataset, tokenizer, max_length=512):
        #TODO: add code to generate tokenized version of the dataset
        tokenized_dataset = []
        return tokenized_dataset


    def get_dataloader(tokenized_dataset, batch_size, shuffle=False):
        return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=shuffle)


    def cosine_sim(a, b):
        # TODO: Implement cosine similarity function **from scrach**:
        # This method should expect two 2D matrices (batch, vector_dim) and
        # return a 2D matrix (batch, batch) that contains all pairwise cosine similarities
        return torch.zeros(a.shape[0], a.shape[0])


    def eval_loop(model, eval_dataloader, device):
        #TODO: add code to for evaluation loop
        #TODO: Use cosine_sim function above as distance metric for pearsonr and spearmanr functions that are imported
        return [eval_pearson_cosine, eval_spearman_cosine]


    #INFO: model and tokenizer
    model_name = 'prajjwal1/bert-tiny'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #INFO: load bert
    bert_config = {"hidden_size": 128, "num_attention_heads": 2, "num_hidden_layers": 2, "intermediate_size": 512, "vocab_size": 30522}
    bert = Bert(bert_config).load_model('bert_tiny.bin')

    #INFO: load dataset
    sts_dataset = load_sts_dataset('stsbenchmark.tsv.gz')

    #INFO: tokenize dataset
    tokenized_test = tokenize_sentence_pair_dataset(sts_dataset['test'], tokenizer)

    #INFO: generate dataloader
    test_dataloader = get_dataloader(tokenized_test, batch_size=1)

    #INFO: run evaluation loop
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    results_from_pretrained = eval_loop(bert, test_dataloader, device)

    print(f'\nPearson correlation: {results_from_pretrained[0]:.2f}\nSpearman correlation: {results_from_pretrained[1]:.2f}')