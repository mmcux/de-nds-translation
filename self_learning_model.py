# update from mac
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Autodiactic Dataselection Model

# As introduced in the data_preprocessing notebook, we have a lot of wrong aligned sentences in the wikipedia-dataset.
# Goal of this notebook is to clean the wrong sentences as much as possible and create a good databasis for future models.

# We will work with PyTorch and Torchtext. The basis construction of the model is taken from [Bent Revett](https://github.com/bentrevett/pytorch-seq2seq) who builded a NMT-Transformer Seq2Seq similar to the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
# This paper from Google marks the change in State-of-the-Art models in machine translation from RNN and CNN's to attention-transformer models.
# %%
import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.data import Field, BucketIterator,  TabularDataset
from torchtext.data.functional import generate_sp_model, load_sp_model, sentencepiece_numericalizer, sentencepiece_tokenizer
from nltk.translate import bleu_score


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
import pandas as pd
import dill
from sklearn.model_selection import train_test_split

from pathlib import Path
import os
import random
import re
import math
import time

# %% [markdown]

# # Seed defintion for reproducable results

# %%
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# lets define already our device and make sure to run on gpu if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %% [markdown]
# We use the same tokenizer as in the data-preprocessing line



# %%
sp_deu = load_sp_model("preprocessed_data/sp_model/de.wiki.bpe.vs10000.model")
sp_nds = load_sp_model("preprocessed_data/sp_model/nds.wiki.bpe.vs10000.model")


# %%

sp_deu_tokens_generator = sentencepiece_tokenizer(sp_deu)
list_a = ["Komplizierte Wörter sind Baustelle.", "Morgen soll es regnen und übermorgen scheint die Sonne"]
print(list(sp_deu_tokens_generator(list_a)))
sp_numericalize_generator = sentencepiece_numericalizer(sp_deu)
print(list(sp_numericalize_generator(list_a)))

# %%

sp_deu_tokens_generator = sentencepiece_tokenizer(sp_deu)
sp_nds_tokens_generator = sentencepiece_tokenizer(sp_nds)


def tokenize_de(text):
    return list(sp_deu_tokens_generator([text]))[0]
def tokenize_nds(text):
    return list(sp_nds_tokens_generator([text]))[0]

# %% [markdown]
#  In each round we have new training and validation data. We need a function which creates for each round new fields and bucket iterators. With the fields we create in each round a new vocabulary.
# We set the min frequency in the vocabulary to 1. We have a lot of words which appear only once as we saw in the data preprocessing step. If we would set it to two, we probably miss in most sentences the meaning.
# Moreover we set the batch-size to 64 as it is better to calculate.
# %%
# loading in the data into torchtext datasets
def load_train_test_data(path):
    train_path = path / "train_data.csv"
    valid_path = path / "valid_data.csv"
    SRC = Field(tokenize = tokenize_de, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = False, 
            batch_first = True)

    TRG = Field(tokenize = tokenize_nds, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = False, 
            batch_first = True)

    train_data = TabularDataset(path=train_path, format= "csv", skip_header = True
                            , fields = [('id', None),("src", SRC),("trg", TRG)])
    valid_data = TabularDataset(path=valid_path, format= "csv", skip_header = True
                            , fields = [('id', None),("src", SRC),("trg", TRG)])

    SRC.build_vocab(train_data, min_freq = 1)
    TRG.build_vocab(train_data, min_freq = 1)

    BATCH_SIZE = 64

    train_iterator, valid_iterator = BucketIterator.splits(
        (train_data, valid_data), 
        batch_size = BATCH_SIZE,
        sort_within_batch = True,
        sort_key = lambda x : len(x.src),
        device = device)
    return SRC, TRG, train_iterator, valid_iterator


# %% [markdown]

# From here on we define the classes according to the tutorial.
# %%
class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = 100):
        super().__init__()

        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len]
        #src_mask = [batch size, src len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [batch size, src len]
        
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        
        #src = [batch size, src len, hid dim]
        
        for layer in self.layers:
            src = layer(src, src_mask)
            
        #src = [batch size, src len, hid dim]
            
        return src

# 
# %%
class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, src len]
                
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src


# %%
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention

# %%
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x


# %%
class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = 100):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, trg len]
        #src_mask = [batch size, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        #pos = [batch size, trg len]
            
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
                
        #trg = [batch size, trg len, hid dim]
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        output = self.fc_out(trg)
        
        #output = [batch size, trg len, output dim]
            
        return output, attention


# %%
class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, trg len]
        #src_mask = [batch size, src len]
        
        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        #trg = [batch size, trg len, hid dim]
            
        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        #trg = [batch size, trg len, hid dim]
        
        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return trg, attention

class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention

# %% [markdown]
# For each round we stay with the same model. Still we need to adapt the input and output dimensions as we have different vocabularies in each round.
# Therefore we build our encoder and decoder each round from scratch.

# %%
def instantiate_objects(SRC,TRG):
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    enc = Encoder(INPUT_DIM, 
                HID_DIM, 
                ENC_LAYERS, 
                ENC_HEADS, 
                ENC_PF_DIM, 
                ENC_DROPOUT, 
                device)

    dec = Decoder(OUTPUT_DIM, 
                HID_DIM, 
                DEC_LAYERS, 
                DEC_HEADS, 
                DEC_PF_DIM, 
                DEC_DROPOUT, 
                device)
    return enc, dec




# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#print(f'The model has {count_parameters(model):,} trainable parameters')


# %%
# function to initalize the weights of our net
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


# %%


# %% [markdown]
# training and evaluation is exactly as proposed from Bent.
# %%
def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output, _ = model(src, trg[:,:-1])
                
        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]
            
        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
                
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
            
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

# %%
def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output, _ = model(src, trg[:,:-1])
            
            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]
            
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            
            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]
            
            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

# %% [markdown]
# We then define a small function that we can use to tell us how long an epoch takes.

# %%
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# %% [markdown]
# Now we define a translation function and a function for calculating the BLEU-Score.
# Different than in the proposal from Bent we use the Bleu Score from NTLK package so it runs without problems in Colab. 

# %%

def calculate_bleu(data, src_field, trg_field, model, device, max_len = 50):
    
    trgs = []
    pred_trgs = []
    
    for datum in data:
        
        src = vars(datum)['src']
        trg = vars(datum)['trg']
        
        pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len)
        
        #cut off <eos> token
        pred_trg = pred_trg[:-1]
        
        pred_trgs.append(pred_trg)
        trgs.append([trg])
        
    return bleu_score.corpus_bleu(trgs, pred_trgs)

def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50):
    
    model.eval()
        
    if isinstance(sentence, str):
        tokens = tokenize_de(sentence)
    else:
        tokens = [token for token in sentence]


    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:], attention

# %% [markdown]

# In each round we create new training data. We could store just the indices, but as it is easier to load a text file into TorchText than making a workaround for a dataframe, we save it.
# Moreover it is easier to share and get the data of only one round later.

# %%
# saving the first train_test_split
def save_train_test_split(df, path):
    try:
        # Create target Directory
        os.makedirs(path)
        print("Directory " , path ,  " Created ") 
    except FileExistsError:
        print("Directory " , path ,  " already exists") 
    train_data, valid_data = train_test_split(df, test_size=0.1, random_state=SEED)

    train_data.to_csv(path_or_buf= path / "train_data.csv")
    valid_data.to_csv(path_or_buf= path / "valid_data.csv")

    print("Numbers of training samples: " , len(train_data))
    print("Number of validation samples: ",len(valid_data))



# read train-test-split

def read_train_test_split(path):
    train_df = pd.read_csv(path / "train_data.csv", index_col=0)
    valid_df = pd.read_csv(path / "valid_data.csv", index_col=0)
    dataset = train_df.append(valid_df)
    return dataset


# %%

# creating and saving the residuals

def save_residual_data(df, path):
    df.to_csv(path / "residuals.tsv", sep="\t")

# %% [markdown]

# The idea is to evaluate the data which is not in the train-dataset yet. For that we need to store the loss of every sentence pair and return it.
# %%
# calculate residual_loss
def evaluate_residual(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    batch_loss = np.zeros(len(iterator))
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output, _ = model(src, trg[:,: - 1])
            
            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]
            
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            
            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]
            
            loss = criterion(output, trg)
            batch_loss[i] = loss.item()
            epoch_loss += loss.item()
            

        
    return epoch_loss / len(iterator), batch_loss


# %% [markdown]

# Now we will load in our preprocessed data which we created in the data_preprocessing notebook.

# Here you can choose if you want the dataset with a higher variety of spelling in Low German or the more uniform dataset.

# %%

tatoeba_df = pd.read_csv("preprocessed_data/tatoeba/tatoeba_dataset_cleaned_spelling.csv", index_col = 0)
wiki_df = pd.read_csv("preprocessed_data/fb-wiki/wiki_dataset_cleaned_spelling.csv", index_col = 0)



#%%
# loading seperate test set and delete it from dataset
timestr = time.strftime("%Y%m%d-%H%M%S")
base_path = Path("data_selection/")
path = base_path / timestr 

        # Create target Directory
try:       
    os.makedirs(path)
    print("Directory " , path ,  " Created ") 
except FileExistsError:
    print("Directory " , path ,  " already exists") 


# As already prepared in the preprocessing_data notebook, we load in our test_data
# in round seven we had in a previous run more or less an equal test set
# means ca. 50% of the test set are from tatoeba and the other 50% from Wikipedia
test_df = pd.read_csv("preprocessed_data/preprocessed_test_data.csv", index_col=0)

# looks a bit complicated but this way we get the right index of the test-data
# independent from the former index
delete_from_tatoeba = tatoeba_df.reset_index().merge(test_df, on = ["deu","nds"]).set_index("index").index
delete_from_wiki = wiki_df.reset_index().merge(test_df, on = ["deu","nds"]).set_index("index").index

print("Dropping test entries from tatoeba: ", len(delete_from_tatoeba))
print("Dropping test entries from Wikipedia: ", len(delete_from_wiki))
tatoeba_df.drop(delete_from_tatoeba, inplace=True)
wiki_df.drop(delete_from_wiki, inplace=True)

test_path = path / "test_data.csv"
test_df.to_csv(test_path)

#%%
# Per chance I got a sample that is in Low German two times. Once in the train data and once in the test-data but with different German sentences
# but deleting worked as other examples prooved. 

test_string = test_df.sample(1, random_state = SEED).nds.tolist()[0]
print(test_df.sample(1, random_state = SEED))
print(tatoeba_df[tatoeba_df.nds.str.contains(test_string)])
print(wiki_df[wiki_df.nds.str.contains(test_string)])

test_string = test_df.sample(1, random_state = 42).nds.tolist()[0]
print(test_df.sample(1, random_state = 42))
print(tatoeba_df[tatoeba_df.nds.str.contains(test_string)])
print(wiki_df[wiki_df.nds.str.contains(test_string)])

#%%
# creating data for the basis round

runs = 14

path_round = path / "round_"

# the first round is completed with the tatoeba dataset
save_train_test_split(tatoeba_df, path_round / str(0) )



# %%
# creating dataframes for collecting results
loss_summary = pd.DataFrame(np.zeros([len(wiki_df),runs]), index = wiki_df.index)


round_stats = pd.DataFrame(columns = ["best_valid_loss", "epoch_mins", "epoch_secs", "test_loss",
                                       "residual_loss","residual_mins","residual_secs","quantile",
                                       "test_bleu", "total_samples"])
# %%

# define error quantile until which the data should be kept for the next round 
#quantile = 0.25
include_bleu = False

residual_loss_before = float("Inf")



# %%


for i in range(runs):

    print("===================================================")
    print("Round: ", i)
    path_iter = path_round / str(i) 
    # load the iterators which contains already the batches
    SRC, TRG, train_iterator, valid_iterator = load_train_test_data(path_iter)
    torch.save(SRC, path_iter / "SRC.Field", pickle_module = dill)
    torch.save(TRG, path_iter / "TRG.Field", pickle_module = dill)
    # count how many samples we have int total
    total_samples = (len(train_iterator) + len(valid_iterator))*64
    test_data = TabularDataset(path=test_path, format= "csv", skip_header = True
                        , fields = [('id', None),("src", SRC),("trg", TRG)])

    test_iterator = BucketIterator(
        test_data, 
        batch_size = 64,
        sort_within_batch = True,
        sort_key = lambda x : len(x.src),
        device = device)

    preload_model = True
    if preload_model == False:
        enc , dec = instantiate_objects(SRC,TRG)

        SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
        TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
        print(TRG_PAD_IDX)
        model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
        # random weigths
        model.apply(initialize_weights)
    else:
        path_pretrained = Path("model/pre_training/20200623-132115_CAPITAL_MINFREQ2_BATCH32_SAMETOKENIZER_NLD_PRETRAINING_DE_EN/")
        path_pretrained = Path("model/pre_training/20200707-090432_CAPITAL_MINFREQ1_BATCH32_SAMETOKENIZER_EN_SENTENCEPIECE_DE_EN/")
        #path_pretrained = "model/sentencepiece/20200707-111219_CAPITAL_MINFREQ1_BATCH32_SAMETOKENIZER_EN_SENTENCEPIECE_10k_DE_NDS/"
        #path_pretrained = "model/pre_training/20200708-124955_CAPITAL_MINFREQ1_BATCH32_SAMETOKENIZER_NL_10k_SENTENCEPIECE_10k_DE_NL/"

        pre_SRC = torch.load(path_pretrained / "SRC.Field", pickle_module=dill)
        pre_TRG = torch.load(path_pretrained / "TRG.Field", pickle_module=dill)
        enc , dec = instantiate_objects(SRC,TRG)

        SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
        TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

        model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
        model.apply(initialize_weights)
        model_dict = model.state_dict()
        # loaded weights from pretrained model
        pretrain_dict = torch.load(path_pretrained / 'model.pt')
        #Filter out unnecessary keys
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if (k in model_dict and 'fc_out' not in k and 'tok_embedding' not in k)}
        model.load_state_dict(pretrain_dict, strict=False)
        #model.load_state_dict(torch.load(path_pretrained + 'model.pt', map_location=torch.device(device)),strict=False)
        print("Loaded pretrained model")

    LEARNING_RATE = 0.0005
    CLIP = 1


    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)



    N_EPOCHS = 6

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        

        start_time = time.time()
        
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), path_iter / 'model.pt')
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')



    model.load_state_dict(torch.load(path_iter / 'model.pt'))

    test_loss = evaluate(model, test_iterator, criterion)

    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')


    # calculating new size of dataset
    # we want to start small and get bigger over time
    # our model should get better and can predict better for larger datasets, which sentences are good
    # we start with 500 and increase 
    residual_size =  500 * 2**(i)
    
    start_index = wiki_df.index[0]
    # if the calculated size exceeds the wiki_df, we take the full remaining dataframe
    if len(wiki_df) > residual_size:
      end_index = wiki_df.index[residual_size]
    else:
      end_index = wiki_df.index[-1]
    residual_df = wiki_df.loc[start_index:end_index,:].copy()
    residual_df.to_csv(path_iter / "residuals.tsv", sep="\t")

    # calculating the error for the data which was not included in the model & testing
    residual_pairs = TabularDataset(path=path_iter / "residuals.tsv", format= "tsv", skip_header = True
                                , fields = [('id', None),("src", SRC),("trg", TRG)])
    debug_text_residual_src = vars(residual_pairs.examples[8])['src']
    debug_text_residual_trg = vars(residual_pairs.examples[8])['trg']
    print("Debug Residual Text Source: ", debug_text_residual_src)
    print("Debug Residual Text Target: ", debug_text_residual_trg)


    residual_iterator = BucketIterator(residual_pairs, batch_size = 1, device = device
                                   , shuffle = False , sort_within_batch=False , repeat = False)

    start_time = time.time()
    
    residual_loss , residual_batch_loss = evaluate_residual(model, residual_iterator, criterion)

    end_time = time.time()
    residual_mins, residual_secs = epoch_time(start_time, end_time)
    print("Residual loss total: ",residual_loss)
    print(f"Residual Evaluation Time: {residual_mins}m {residual_secs}s" )
    # appending the error to our data and select only the best 25%
    #residual_df = pd.read_csv(path_iter + "residuals.tsv", sep="\t", index_col=0)
    residual_df.loc[:,"loss"] = residual_batch_loss

    #storing the loss in the loss summary at the right sentence index

    loss_summary.iloc[:, i] = residual_df.loss
    loss_summary.to_csv(path / "loss_summary.csv")

    # calcualting the 25% quantile
    quantile = residual_df.loss.quantile(0.25)
    print("Quantile: ", quantile)

    # appending the best 25% to our existing training dataset and split & save for next round
    new_train_data = residual_df[residual_df.loss <= quantile][["deu","nds"]]
    old_train_data = read_train_test_split(path_iter)
    dataset = old_train_data.append(new_train_data)

    new_path = path_round / str(i + 1) 

    # shuffling for the next round is important, so the new dataset is integrated through the whole training process
    # it is done inside the below function before saving it
    save_train_test_split(dataset, new_path)

    # save the new train-data for easy quality check
    new_train_data.to_csv(new_path / "new_training_data.csv")

    # drop the new included train sentences from the wiki_df
    # so they are excluded for next rounds
    wiki_df = wiki_df.drop(new_train_data.index)



    # calculating bleu score
    if include_bleu == True:
        test_bleu = calculate_bleu(test_data, SRC, TRG, model, device)
        print("Test BLEU-Score: ",test_bleu)
        
    else:
        test_bleu = np.NaN


    # saving stats
    round_stats.loc[i, :] = [best_valid_loss, epoch_mins, epoch_secs, test_loss,residual_loss,
                    residual_mins,residual_secs,quantile, test_bleu, total_samples]
    round_stats.to_csv(path / "round_stats.csv")








# %%


# %% [markdown]
# for quick checking if the model is ok, we can try out some sample sentences
# %%
def display_attention(sentence, translation, attention, n_heads = 8, n_rows = 4, n_cols = 2):
    
    assert n_rows * n_cols == n_heads
    
    fig = plt.figure(figsize=(15,25))
    
    for i in range(n_heads):
        
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], 
                           rotation=45)
        ax.set_yticklabels(['']+translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()





# # %% [markdown]
# # Finally, we'll look at an example from the test data.

# %%
example_idx = 2132

src = vars(test_data.examples[example_idx])['src']
trg = vars(test_data.examples[example_idx])['trg']

print(f'src = {src}')
print(f'trg = {trg}')

# # %% [markdown]
# # The translation from test dataset is also correct.

# # %%
translation, attention = translate_sentence(src, SRC, TRG, model, device)

print(f'predicted trg = {translation}')


# # %%
display_attention(src, translation, attention)



# %%
bleu_score = calculate_bleu(test_data, SRC, TRG, model, device)

print(f'BLEU score = {bleu_score*100:.2f}')

# %% [markdown]
# In this section you can enter a custom sentence and look at the predicted translation.

# %%
custom_sentence = "Plattdeutsch auf Wikipedia ist eine Katastrophe."

translation, attention = translate_sentence(custom_sentence, SRC, TRG, model, device)
print("In German: ", custom_sentence)
print("In Low German: ", ' '.join(translation[:-1]))


# %%



