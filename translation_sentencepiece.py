
# %%
import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.data import Field, BucketIterator,  TabularDataset
from torchtext.data.functional import generate_sp_model, load_sp_model, sentencepiece_numericalizer, sentencepiece_tokenizer
from nltk.translate import bleu_score
import sentencepiece as spm

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import dill
import numpy as np
import pandas as pd

from pathlib import Path
import os
import random
import re
import math
import time


# %%

SEED = 1234

#torch.cuda.set_device(0)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

print("Runs on Cuda: ", torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
# creating data for the basis round
# select the run which should be the basis
run = 9


timestr = time.strftime("%Y%m%d-%H%M%S")
base_path = Path("model/sentencepiece/")
saving_path =  base_path / timestr / "english_pretraining_with_sp/"
# saving the model and results
try:
    # Create target Directory
    os.makedirs(saving_path)
    print("Directory " , saving_path ,  " Created ") 
except FileExistsError:
    print("Directory " , saving_path ,  " already exists") 



local_data = True

if local_data:
    path = "preprocessed_data/tatoeba/"
    path = Path("preprocessed_data/english_training/")
    #path = "preprocessed_data/dutch_training/"
    #path = "model/finetuned_after_pre_training/20200620-185343_CAPITAL_MINFREQ2_BATCH32_SAMETOKENIZER_DE_DE_/"
    path = Path("data_selection/20200723-165133/round_/8/")
    path_round = path / "round_"
    train_path = path / "train_data.csv"
    valid_path = path / "valid_data.csv"
    test_path = path  / "test_data.csv"
else:
    # your link here
    # the following links include SELF-LEARNING DATA in train and valid
    train = "https://llmm.webo.family/index.php/s/Ra7bT3QPcfRbXA8/download"
    valid = "https://llmm.webo.family/index.php/s/i7FcJDSbX3i5Esk/download"
    test = "https://llmm.webo.family/index.php/s/AaBzT6kZnAp2o9C/download"
    # we have to save our data locally
    train_csv = pd.read_csv(train, index_col = 0)
    valid_csv = pd.read_csv(valid, index_col = 0)
    test_csv = pd.read_csv(test, index_col = 0)
    train_path = saving_path / "train_data.csv"
    train_csv.to_csv(train_path)
    valid_path = saving_path / "valid_data.csv"
    valid_csv.to_csv(valid_path)
    test_path = saving_path / "test_data.csv"
    test_csv.to_csv(test_path)

# %%

# add additional data only for train set to see if it performs better
additional_data = True
if additional_data:
    train = pd.read_csv(train_path, index_col = 0)
    wordbook = pd.read_csv("preprocessed_data/hansen/wordbook.csv", index_col = 0)
    wordbook = wordbook[["deu", "nds"]]
    print(wordbook.head(3))
    # right now only wordbook as GPU storage is too small for both
    train = train.append(wordbook).sample(frac=1).reset_index(drop=True)
    print(train.head(5))
    train.to_csv(saving_path / "train_data.csv")
    train_path = saving_path / "train_data.csv"
    print(str(train_path))


# %%
vocab_size = 10000

train_sp = pd.read_csv(train_path)
# create solo datasets for german and low german to prepare for training sentencepiece model
deu_sentences = train_sp.iloc[:,1]
deu_sentences_path = saving_path /  "deu_train_sentences.csv"
deu_sentences.to_csv(deu_sentences_path, index=False, header = False)
nds_sentences = train_sp.iloc[:,2]
nds_sentences_path = saving_path /  "nds_train_sentences.csv"
nds_sentences.to_csv(nds_sentences_path, index=False, header = False)

# define that padding_idx is on 3 otherwise it wouldn't be defined
def spm_args(data_path, model_prefix, vocab_size ):
    return ''.join(["--pad_id=3",' --input=', str(data_path)," --model_prefix=", model_prefix, " --vocab_size=",str(vocab_size)])
# create input string and train sentencepiecemodel
deu_spm_input = spm_args(deu_sentences_path, str(saving_path / "spm_deu"), vocab_size)
nds_spm_input = spm_args(nds_sentences_path, str(saving_path / "spm_nds"), vocab_size)
spm.SentencePieceTrainer.train(deu_spm_input)
spm.SentencePieceTrainer.train(nds_spm_input)

#%%

sp_deu = load_sp_model(str(saving_path / "spm_deu.model"))
sp_nds = load_sp_model(str(saving_path / "spm_nds.model"))




#%%
# add monolingual data only here as it would else included into sentencepiece model
mono_data = True
if mono_data:
    mono = pd.read_csv("preprocessed_data/monolingual.csv", index_col = 0)
    mono = mono[["deu", "nds"]]
    mono = mono.sample(100000, random_state=42)
    mono_enc = mono.deu.apply(sp_deu.encode)
    mono_len = mono_enc.apply(len)
    mono_drop = mono_len[mono_len >= 60].index
    print("Drops from mono due to length: ", len(mono_drop))
    mono.drop(mono_drop, inplace = True)
    print(mono.head(4))
    train = train.append(mono).sample(frac=1).reset_index(drop=True)
    print(train.head(5))

    train.to_csv(saving_path / "train_data.csv")
    train_path = saving_path / "train_data.csv"    
# %%



# %%

# loading in the data into torchtext datasets



SRC = Field(use_vocab = False, tokenize = sp_deu.encode,
            init_token = sp_deu.bos_id(), 
            eos_token = sp_deu.eos_id(),
            pad_token = sp_deu.pad_id(),
            batch_first = True
            )

TRG = Field(use_vocab = False, tokenize = sp_nds.encode,
            init_token = sp_nds.bos_id(), 
            eos_token = sp_nds.eos_id(),
            pad_token = sp_nds.pad_id(),
            batch_first = True
            )

train_data = TabularDataset(path=train_path, format= "csv", skip_header = True
                        , fields = [('id', None),("src", SRC),("trg", TRG)])
valid_data = TabularDataset(path=valid_path, format= "csv", skip_header = True
                        , fields = [('id', None),("src", SRC),("trg", TRG)])
test_data = TabularDataset(path= test_path, format= "csv", skip_header = True
                    , fields = [('id', None),("src", SRC),("trg", TRG)])


# %% 
BATCH_SIZE = 32


test_iterator = BucketIterator(
    test_data, 
    batch_size = BATCH_SIZE,
    sort_within_batch = True,
    sort_key = lambda x : len(x.src),
    device = device)




train_iterator, valid_iterator = BucketIterator.splits(
    (train_data, valid_data), 
    batch_size = BATCH_SIZE,
    sort_within_batch = True,
    sort_key = lambda x : len(x.src),
    device = device)


torch.save(SRC, saving_path / "SRC.Field", pickle_module = dill)

torch.save(TRG, saving_path / "TRG.Field", pickle_module = dill)

# %%
# check if everything worked out with sentencepiece
example_id = 234
example_deu = vars(train_data.examples[example_id])['src']
print(example_deu)
print(sp_deu.decode(example_deu))
example_nds = vars(train_data.examples[example_id])['trg']
print(example_nds)
print(sp_nds.decode(example_nds))
len(sp_nds)

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

"""### Seq2Seq"""

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

"""
## Training the Seq2Seq Model

 We can now define our encoder and decoders. This model is significantly smaller than Transformers used in research today, but is able to be run on a single GPU quickly.
"""

def instantiate_objects(SRC,TRG, vocab_size=vocab_size):
    INPUT_DIM = vocab_size
    OUTPUT_DIM = vocab_size
    # hidden_dim was 256 before
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

"""Then, use them to define our whole sequence-to-sequence encapsulating model.

We can check the number of parameters, noticing it is significantly less than the 37M for the convolutional sequence-to-sequence model.
"""

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#print(f'The model has {count_parameters(model):,} trainable parameters')

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

"""The optimizer used in the original Transformer paper uses Adam with a learning rate that has a "warm-up" and then a "cool-down" period. BERT and other Transformer models use Adam with a fixed learning rate, so we will implement that. Check [this](http://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer) link for more details about the original Transformer's learning rate schedule.

 Note that the learning rate needs to be lower than the default used by Adam or else learning is unstable.
"""



"""Next, we define our loss function, making sure to ignore losses calculated over `<pad>` tokens."""



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

"""The evaluation loop is the same as the training loop, just without the gradient calculations and parameter updates."""

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

"""We then define a small function that we can use to tell us how long an epoch takes."""

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def calculate_bleu(data, src_field, trg_field, model, device, max_len = 50):
    
    trgs = []
    pred_trgs = []
    
    for datum in data:
        
        src = vars(datum)['src']
        trg = vars(datum)['trg']
        
        pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len)
        
        #cut off <eos> token
        #pred_trg = pred_trg[:-1]
        
        pred_trgs.append(pred_trg)
        trgs.append([sp_nds.decode(trg)])
        
    return bleu_score.corpus_bleu(trgs, pred_trgs)

def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50):
    
    model.eval()
        
    if isinstance(sentence, str):
        tokens = sp_deu.encode(sentence)

    else:
        tokens = [token for token in sentence]

    if tokens[0] != src_field.init_token and tokens[-1] != src_field.eos_token:
        tokens = [src_field.init_token] + tokens + [src_field.eos_token]


    src_indexes = tokens
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.init_token]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.eos_token:
            break

    trg_tokens = sp_nds.decode(trg_indexes)
    
    return trg_tokens, attention

# %%

# creating dataframes for collecting results

round_stats = pd.DataFrame(columns = ["best_valid_loss", "epoch_mins", "epoch_secs", "test_loss",
                                       "test_bleu", "total_samples"])

include_bleu = True

residual_loss_before = float("Inf")

print("===================================================")
print("Round: ", run)
path_iter = saving_path
# count how many samples we have int total
total_samples = (len(train_iterator) + len(valid_iterator))*BATCH_SIZE

print("All data read in")

preload_model = True
if preload_model == False:
    enc , dec = instantiate_objects(SRC,TRG)

    SRC_PAD_IDX = SRC.pad_token
    TRG_PAD_IDX = TRG.pad_token
    print(type(TRG_PAD_IDX))
    print(SRC_PAD_IDX)
    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    # random weigths
    model.apply(initialize_weights)
else:
    path_pretrained = "model/pre_training/20200623-132115_CAPITAL_MINFREQ2_BATCH32_SAMETOKENIZER_NLD_PRETRAINING_DE_EN/"
    path_pretrained = Path("model/pre_training/20200707-090432_CAPITAL_MINFREQ1_BATCH32_SAMETOKENIZER_EN_SENTENCEPIECE_DE_EN/")
    path_pretrained = Path("model/pre_training/20200727-212428/english_with_sp/")
    #path_pretrained = Path("model/sentencepiece/20200727-224225/english_pretraining_with_sp/")
    #path_pretrained = "model/sentencepiece/20200707-111219_CAPITAL_MINFREQ1_BATCH32_SAMETOKENIZER_EN_SENTENCEPIECE_10k_DE_NDS/"
    #path_pretrained = "model/pre_training/20200708-124955_CAPITAL_MINFREQ1_BATCH32_SAMETOKENIZER_NL_10k_SENTENCEPIECE_10k_DE_NL/"

    pre_SRC = torch.load(path_pretrained / "SRC.Field", pickle_module=dill)
    pre_TRG = torch.load(path_pretrained / "TRG.Field", pickle_module=dill)
    enc , dec = instantiate_objects(SRC,TRG)

    SRC_PAD_IDX = SRC.pad_token
    TRG_PAD_IDX = TRG.pad_token

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



# %%

N_EPOCHS = 6

best_valid_loss = float('inf')
print("Starting calculating epochs")
for epoch in range(N_EPOCHS):
    

    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), saving_path / 'model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')



model.load_state_dict(torch.load(saving_path / 'model.pt'))

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')


# calculating bleu score
if include_bleu == True:
    test_bleu = calculate_bleu(test_data, SRC, TRG, model, device)
    print("Test BLEU-Score: ",test_bleu)
    
else:
    test_bleu = np.NaN


# saving stats
round_stats.loc[run, :] = [best_valid_loss, epoch_mins, epoch_secs, test_loss, test_bleu, total_samples]
round_stats.to_csv(saving_path / "round_stats.csv")

#%%





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
example_idx = 31

src = vars(test_data.examples[example_idx])['src']
trg = vars(test_data.examples[example_idx])['trg']

print(f'src = {sp_deu.decode(src)}')
print(f'trg = {sp_nds.decode(trg)}')

# # %% [markdown]
# # The translation from test dataset is also correct.

translation, attention = translate_sentence(src, SRC, TRG, model, device)

print(f'predicted trg = {translation}')


# %%

ls = [1, 117, 1053, 3943, 593, 8589, 2]
if ls[0] == 1 and ls[-1] == 2:
    print("ok")
# %%
#display_attention(src, translation, attention)


# %%

"""In this section you can enter a custom sentence and look at the predicted translation."""

custom_sentence = "Die Schranke ist geschlossen."

translation, attention = translate_sentence(custom_sentence, SRC, TRG, model, device)
print("In German: ", custom_sentence)
#print("In Low German: ", ''.join(translation[:-1]).replace('▁', ' '))
print("In Low German: ", translation)

# %%


# %%
