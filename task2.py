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


def load_sts_dataset(file_name):
    # TODO: add code to load STS dataset in required format
    with gzip.open(file_name, 'rb') as f:
        content = f.read().decode('utf-8')

    l_line = [line.split('\t') for line in content.split('\r\n') if '\t' in line]
    df = pd.DataFrame(data=l_line[1:], columns=l_line[0], dtype=object)

    df['score'] = pd.Series(data=[np.float32(score) for score in df['score']], dtype=np.float32, index=df.index)

    sts_samples = {split: df.loc[df['split'].values == split] for split in np.unique(df['split'])}

    return sts_samples


def tokenize_sentence_pair_dataset(dataset, tokenizer, max_length=512):
    # TODO: add code to generate tokenized version of the dataset

    tokenized_dataset = dataset.copy() # df
    for column in ['sentence1', 'sentence2']:
        l_token = []
        for sentence in tokenized_dataset[column].values:
            sentence_token = tokenizer(sentence, return_tensors='pt', padding='max_length', max_length=max_length)
            l_token.append(sentence_token)
        tokenized_dataset[column + '_token'] = pd.Series(data=l_token, index=tokenized_dataset.index, dtype=object)

    return tokenized_dataset


def get_dataloader(tokenized_dataset, batch_size, shuffle=False):
    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=shuffle)


def cosine_sim(a, b):
    # TODO: Implement cosine similarity function **from scrach**:
    # This method should expect two 2D matrices (batch, vector_dim) and
    # return a 2D matrix (batch, batch) that contains all pairwise cosine similarities
    return torch.zeros(a.shape[0], a.shape[0])


def eval_loop(model, eval_dataloader, device):
    # TODO: add code to for evaluation loop
    # TODO: Use cosine_sim function above as distance metric for pearsonr and spearmanr functions that are imported
    # return [eval_pearson_cosine, eval_spearman_cosine]

    
    model.eval()
    model.to(device)

    for data in eval_dataloader:
        data = data.to(device)

        output = model(data)


    return [1, 2]  # fake data


if __name__ == "__main__":
    #INFO: model and tokenizer
    model_name = 'prajjwal1/bert-tiny'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #INFO: load bert
    # bert_config = {"hidden_size": 128, "num_attention_heads": 2, "num_hidden_layers": 2, "intermediate_size": 512, "vocab_size": 30522}
    # bert = Bert(bert_config).load_model('bert_tiny.bin')

    bert = AutoModel.from_pretrained(model_name)

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