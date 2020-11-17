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
from torch.utils.data import DataLoader, Dataset


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pathlib import Path
import os
import random
import re
import math
import time
from transformers import MarianMTModel, MarianTokenizer, AdamW

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


# %%
# We use the same tokenizer as in the data-preprocessing line

model_name = 'Helsinki-NLP/opus-mt-NORTH_EU-NORTH_EU'

tokenizer = MarianTokenizer.from_pretrained(model_name)

# %% [markdown]
#  In each round we have new training and validation data. We need a function which creates for each round new fields and bucket iterators. With the fields we create in each round a new vocabulary.
# %%

class NDSDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        #self.attention_mask = attention_mask
        #self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # item["attention_mask"] =  {key: torch.tensor(val[idx]) for key, val in self.attention_mask.items()}
        # item['labels'] =  {key: torch.tensor(val[idx]) for key, val in self.labels.items()}
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def df_to_iterator(df,tokenizer, batch_size):
    df["deu"] = ">>nl<<" + " " + df["deu"]
    ds = tokenizer.prepare_seq2seq_batch(df["deu"].tolist(),df["nds"].tolist())
    ds = NDSDataset(ds)
    iterator = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    return iterator




# loading in the data into torchtext datasets
def load_train_test_data(path):
    train = pd.read_csv(path / "train_data.csv", index_col=0)
    valid = pd.read_csv(path / "valid_data.csv", index_col=0)

    train_iterator = df_to_iterator(train,tokenizer,16)
    valid_iterator = df_to_iterator(train,tokenizer,16)

    return train_iterator, valid_iterator


# %% [markdown]

# %% [markdown]
# For each round we stay with the same model. Still we need to adapt the input and output dimensions as we have different vocabularies in each round.
# Therefore we build our encoder and decoder each round from scratch.

# %%




# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#print(f'The model has {count_parameters(model):,} trainable parameters')


# %%


# %%


# %% [markdown]
# training and evaluation is exactly as proposed from Bent.
# %%
def train(model, iterator, optimizer):
    
    model.train()
    time_col = 0
    epoch_loss = 0
    #training
    for idx,batch in enumerate(iterator):
        start = time.time()
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        end = time.time()
        time_col += end - start
        epoch_loss += loss.item()
        if idx % 1000 == 0:
            print(idx)
            print(f"time {(time_col) / 60 } min")
    return epoch_loss / len(iterator)    

# %%
def evaluate(model, iterator):
    
    model.eval()
    valid_loss = 0
    # evaluation set
    for idx,batch in enumerate(iterator):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        loss = model(input_ids, attention_mask=attention_mask, labels=labels)[0].item()
        valid_loss += loss
    return valid_loss / len(iterator)

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

def translate_sentence(sentence, model,tokenizer, device):
    input_ids = tokenizer(f">>nl<< {sentence}", return_tensors="pt").input_ids

    output_ids = model.generate(input_ids.to(device))

    translation = tokenizer.decode(output_ids[0],skip_special_tokens=True)
    return translation
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
def evaluate_residual(model, iterator):
    
    model.eval()
    
    epoch_loss = 0
    batch_loss = np.zeros(len(iterator))
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            loss = model(input_ids, attention_mask=attention_mask, labels=labels)[0].item()
            
            batch_loss[i] = loss
            epoch_loss += loss
            
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

test_iterator = df_to_iterator(test_df,tokenizer,16)

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
    train_iterator, valid_iterator = load_train_test_data(path_iter)
    # count how many samples we have int total
    total_samples = (len(train_iterator) + len(valid_iterator))*16

    freeze_encoder = True
    if freeze_encoder == False:
        model = MarianMTModel.from_pretrained(model_name)
        model.to(device)
        print("Loaded pretrained model without freezing encoder")
    else:
        model = MarianMTModel.from_pretrained(model_name)
        model.to(device)
        for param in model.base_model.encoder.parameters():
            param.requires_grad = False
        print("Loaded pretrained model with freezed encoder")


    optimizer = AdamW(model.parameters(), lr=5e-5)


    N_EPOCHS = 3

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        

        start_time = time.time()
        
        train_loss = train(model, train_iterator, optimizer)
        valid_loss = evaluate(model, valid_iterator)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            model.save_pretrained(path_iter)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')



    model = MarianMTModel.from_pretrained(path_iter)
    model.to(device)
    test_loss = evaluate(model, test_iterator)

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
    residual_iterator = df_to_iterator(residual_df,tokenizer,1)
    start_time = time.time()
    
    residual_loss , residual_batch_loss = evaluate_residual(model, residual_iterator)

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
        pass
        
    else:
        test_bleu = np.NaN


    # saving stats
    round_stats.loc[i, :] = [best_valid_loss, epoch_mins, epoch_secs, test_loss,residual_loss,
                    residual_mins,residual_secs,quantile, test_bleu, total_samples]
    round_stats.to_csv(path / "round_stats.csv")








# %%
evaluate_residual(model,residual_iterator)

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



