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

def load_nli_dataset(file_name):
    with gzip.open(file_name, 'rb') as f:
        content = f.read().decode('utf-8')

    l_line = [line.split('\t') for line in content.split('\r\n') if '\t' in line]
    df = pd.DataFrame(data=l_line[1:], columns=l_line[0], dtype=object)

    nli_samples = {split: df.loc[df['split'].values == split] for split in np.unique(df['split'])}

    return nli_samples


def tokenize_sentence_pair_dataset(dataset, tokenizer, max_length=512):
    # TODO: add code to generate tokenized version of the dataset

    df_tokenized = dataset.copy() # df
    for column in ['sentence1', 'sentence2']:
        l_token = []
        for sentence in df_tokenized[column].values:
            sentence_token = tokenizer(sentence, return_tensors='pt', padding='max_length', max_length=max_length)
            l_token.append(sentence_token)
        df_tokenized[column + '_token'] = pd.Series(data=l_token, index=df_tokenized.index, dtype=object)

    tensor_1 = torch.from_numpy(np.array([token['input_ids'] for token in df_tokenized['sentence1_token'].values]))
    tensor_2 = torch.from_numpy(np.array([token['input_ids'] for token in df_tokenized['sentence2_token'].values]))

    tokenized_dataset = torch.utils.data.TensorDataset(tensor_1, tensor_2)

    return tokenized_dataset


def get_dataloader(tokenized_dataset, batch_size, shuffle=False):
    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=shuffle)


def cosine_sim(a, b):
    # TODO: Implement cosine similarity function **from scrach**:
    # This method should expect two 2D matrices (batch, vector_dim) and
    # return a 2D matrix (batch, batch) that contains all pairwise cosine similarities

    divident = torch.matmul(a, b.T)
    bla = torch.outer(torch.sum(a ** 2, 1), torch.sum(b ** 2, 1))
    divisor = torch.maximum(bla, torch.ones_like(bla) * 0.001)

    return divident / divisor

# A periodic eval on dev test can be added (validation_dataloader)
def train_loop(model, optimizer, train_dataloader, num_epochs, device):
    #TODO: add code to for training loop
    #TODO: use optimizer, train_dataloader, num_epoch and device for training

    model.train()
    model.to(device)

    for _ in range(num_epochs):
        for data in train_dataloader:
            a = data[0].to(device)
            b = data[1].to(device)

            batch_a, size_a, vec_size_a = a.shape
            batch_b, size_b, vec_size_b = b.shape

            a = a.reshape((batch_a, vec_size_a)) # .type(torch.float32)
            b = b.reshape((batch_b, vec_size_b)) # .type(torch.float32)

            a_kacke = model.bert(a)
            b_kacke = model.bert(b)

            u = a_kacke.pooler_output
            v = b_kacke.pooler_output

            val_sim = cosine_sim(a=u, b=v)

            output = model(u, v, val_sim)

    pass

class Config(object):
    """Configuration class to store the configuration of a BertModel.
    https://blog.csdn.net/ZJRN1027/article/details/103685696
    """
    def __init__(self,
                vocab_size,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=2,
                initializer_range=0.02):

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = dropout_prob
        self.attention_probs_dropout_prob = dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, dict_object):
        config = Config(vocab_size=None)
        for (key, value) in dict_object.items():
            config.__dict__[key] = value
        return config

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)  # bug 0 repaced by -1
        s = (x + u).pow(2).mean(-1, keepdim=True)  # bug 0 repaced by -1
        x = (x + u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertClassifier(nn.Module):
    def __init__(self, config, num_labels, pooling="max"):
        super(BertClassifier, self).__init__()
        # self.bert = BertModel(config)

        model_name = 'prajjwal1/bert-tiny'
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert_pooler = BertPooler(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.pooling = pooling
        assert self.pooling in ["max", "sum", "last"]

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, LayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, batch, global_step=0):
        input_ids, attention_mask, token_type_ids = batch[:3]
        all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)

        if self.pooling == "max":
            output = torch.max(all_encoder_layers[-1], 1)[0]
        elif self.pooling == "sum":
            output = torch.sum(all_encoder_layers[-1], 1)
        elif self.pooling == "last":
            output = pooled_output
        else:
            raise NotImplementedError()

        logits = self.classifier(output)

        if len(batch) == 4:
            loss_fct = nn.CrossEntropyLoss()
            labels = batch[-1]
            assert labels.size() == (logits.size()[0], 1)
            loss = loss_fct(logits, labels[:,0])
            return loss
        elif len(batch) == 3:
            return logits
        else:
            raise NotImplementedError()


if __name__ == "__main__":
    
    file_name = 'AllNLI.tsv.gz'
    check_file = os.path.isfile(file_name)
    if not check_file:
        url = 'https://sbert.net/datasets/AllNLI.tsv.gz'
        filename = wget.download(url)
        
    #INFO: model and training configs
    model_name = 'prajjwal1/bert-tiny'
    num_epochs = 3
    batch_size = 8
    num_labels = 3
    device = torch.device("cpu")
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_config = {"hidden_size": 128, "num_attention_heads": 2, "num_hidden_layers": 2, "intermediate_size": 512, "vocab_size": 30522}
    bert_path = 'bert_tiny.bin'

    #INFO: load nli dataset
    nli_dataset = load_nli_dataset('AllNLI.tsv.gz')

    #INFO: tokenize dataset
    #WARNING: Use only first 50000 samples and maximum sequence length of 128
    tokenized_train = tokenize_sentence_pair_dataset(nli_dataset['train'][:5000], tokenizer, max_length=128)

    #INFO: generate train_dataloader
    train_dataloader = get_dataloader(tokenized_train, batch_size=batch_size, shuffle=True)

    #TODO: Create a BertClassifier with required parameters
    ###    Replace None with required input based on yor implementation
    bert_classifier = BertClassifier(config=Config(bert_config), num_labels=num_labels)

    #INFO: create optimizer and run training loop
    optimizer = AdamW(bert_classifier.parameters(), lr=5e-5)
    train_loop(bert_classifier, optimizer, train_dataloader, num_epochs, device)
