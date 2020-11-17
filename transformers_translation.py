# %%


from transformers import MarianMTModel, MarianTokenizer, T5Tokenizer, T5ForConditionalGeneration
from transformers import AdamW
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pandas as pd
import time
import torch
from torchtext.data import Field, BucketIterator,  TabularDataset

from sklearn.model_selection import train_test_split

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#%%
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


class FTModel:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on GPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU")
    def __init__(self, tokenizer_name, model_name, freeze_encoder = False):

        self.model = MarianMTModel.from_pretrained(model_name)
        self.model.to(FTModel.device)
        self.tokenizer = MarianTokenizer.from_pretrained(tokenizer_name)    
        if freeze_encoder:
            for param in self.model.base_model.encoder.parameters():
                param.requires_grad = False

    def load_data(self, sentence_pairs_path, monolingual_data_path, prefix_deu, prefix_nds, batch_size = 8):
        '''
        Data must have an index column.
        Loads the data into dataframes, shuffles and splits it into train and validation datasets.
        After that transforming to pytorch dataset.
        Monolingual data of the low-resource-language already split into two columns.
        '''
        sentence_pairs = pd.read_csv(sentence_pairs_path, index_col = 0)
        sentence_pairs.rename(columns={"deu": "src","nds":"trg"}, inplace=True)
        mono = pd.read_csv(monolingual_data_path, index_col = 0).sample(5000,random_state=42)
        mono.rename(columns={"deu": "src","nds":"trg"}, inplace=True)
        mono.iloc[:,0] = prefix_deu + " " + mono.iloc[:,0]
        #creating reverse training data
        sentence_pairs_reverse = sentence_pairs.copy()
        sentence_pairs_reverse.rename(columns={"trg": "src","src":"trg"}, inplace=True)
        sentence_pairs["src"] = prefix_deu + " " + sentence_pairs["src"]
        sentence_pairs_reverse["src"] = prefix_nds + " " + sentence_pairs["src"]
        all_data = sentence_pairs.append(sentence_pairs_reverse , ignore_index=True).append(mono , ignore_index=True)
        train, valid = train_test_split(all_data, test_size=0.1, random_state = 42)
        train_dataset = self.tokenizer.prepare_seq2seq_batch(train["src"].tolist(), train["trg"].tolist(), return_tensors="pt")
        valid_dataset = self.tokenizer.prepare_seq2seq_batch(valid["src"].tolist(), valid["trg"].tolist(), return_tensors="pt")
        train_data = NDSDataset(train_dataset)
        self.train_iterator = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

        valid_data = NDSDataset(valid_dataset)
        self.valid_iterator = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True)

    def test(self,sentence_pairs_path, prefix_deu=">>nl<<", prefix_nds=">>de<<"):
        test_df = pd.read_csv(sentence_pairs_path, index_col = 0)
        test_reverse = test_df.copy()
        test_reverse.rename(columns={"deu": "nds","nds":"deu"}, inplace=True)
        test_df["deu"] = prefix_deu + " " + test_df["deu"]
        test_reverse["deu"] = prefix_nds + " " + test_reverse["deu"]
        test_df = test_df.append(test_reverse)
        test_dataset = self.tokenizer.prepare_seq2seq_batch(test_df["deu"].tolist(),test_df["nds"].tolist())
        test_data = NDSDataset(test_dataset)
        test_iterator = DataLoader(test_data, batch_size=8, shuffle=True, drop_last=True)
        test_loss = self.evaluate_model(test_iterator)
        return test_loss
    @classmethod
    def custom_tokenizer(cls, model, model_path, tokenizer_spm_path):
        '''
        NOT TESTED YET.
        Supply your custom sentencepiece model with a shared vocabluary for source and target language.
        Should contain the language prefixes as special tokens. Uses T5 Tokenizer from transformers library.
        '''
        self.model = model.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer(tokenizer_spm_path)
        vocab_size = self.tokenizer.vocab_size
        self.model.base_model.resize_token_embeddings(vocab_size)
        prefix_deu = ">>nl<< "
        prefix_nds = ">>de<< "    
        return cls()


    def train_model(self, save_inbatch):
        self.model.train()
        time_col = 0
        #training
        for idx,batch in enumerate(self.train_iterator):
            start = time.time()
            self.optim.zero_grad()
            input_ids = batch['input_ids'].to(FTModel.device)
            attention_mask = batch['attention_mask'].to(FTModel.device)
            labels = batch['labels'].to(FTModel.device)
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            self.optim.step()
            end = time.time()
            time_col += end - start
            if idx % 1000 == 0:
                print("Batch:", idx)
                print(f"time {(time_col) / 60 } min")
                self.model.save_pretrained(self.storage_path / "inbatch_model")


    def evaluate_model(self, iterator = None):
        self.model.eval()
        valid_loss = 0
        if iterator is None:
            iterator = self.valid_iterator
        # evaluation set
        for idx,batch in enumerate(iterator):
            start = time.time()
            input_ids = batch['input_ids'].to(FTModel.device)
            attention_mask = batch['attention_mask'].to(FTModel.device)
            labels = batch['labels'].to(FTModel.device)
            loss = self.model(input_ids, attention_mask=attention_mask, labels=labels)[0].item()
            valid_loss += loss
        valid_loss = valid_loss / len(iterator)
        return valid_loss

    def train(self, storage_path, epochs = 4, optimizer = None, save_inbatch = False):
        if optimizer is None:
            optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.optim = optimizer
        self.storage_path = storage_path
        # saving the model and results
        try:
            # Create target Directory
            os.makedirs(self.storage_path)
            print("Directory " , self.storage_path ,  " Created ") 
        except FileExistsError:
            print("Directory " , self.storage_path ,  " already exists")        
        #start training
        best_valid_loss = float('inf')
        for epoch in range(epochs):
            print(f"------Epoch: {epoch} --------")
            self.train_model(save_inbatch = save_inbatch)
            valid_loss = self.evaluate_model()
            print(f"valid loss in epoch {epoch}: {valid_loss}")
            if valid_loss < best_valid_loss:
                self.model.save_pretrained(self.storage_path / "mynewmodel")
                print("new best model")
                best_valid_loss = valid_loss            

    def translate(self, text, trg_lang):
        input_ids = self.tokenizer(f"{trg_lang} {text}", return_tensors="pt").input_ids

        output_ids = self.model.generate(input_ids.to(FTModel.device))

        return  self.tokenizer.decode(output_ids[0],skip_special_tokens=True)

#%%

test = FTModel('Helsinki-NLP/opus-mt-NORTH_EU-NORTH_EU', 'Helsinki-NLP/opus-mt-NORTH_EU-NORTH_EU')
#%%

sentence_pairs_path = Path("preprocessed_data/tatoeba/tatoeba_dataset_cleaned_spelling.csv")
monolingual_data_path = Path("preprocessed_data/monolingual.csv")
test.load_data(sentence_pairs_path = sentence_pairs_path, 
            monolingual_data_path=monolingual_data_path, prefix_deu=">>nl<<", prefix_nds=">>de<<"
            , batch_size=4)
#%%
storage_path = Path("model/transformers/classtest/")
test.train(storage_path)

#%%


test_path = Path("preprocessed_data/preprocessed_test_data.csv")
test_loss = test.test(test_path,">>nl<<",">>de<<")
test_loss
#%%

test.translate("Mit 17 wurde Walter Ansorge von der Mafia entführt. Heute ist der Deutsche einer der erfolgreichsten Unternehmer Siziliens. Und lässt sich nicht mehr einschüchtern.", ">>nl<<")


#%%
custom_tokenizer = False
if custom_tokenizer == True:
    from tokenizers import SentencePieceBPETokenizer
    vocab_size = 20000
    sp_tokenizer = SentencePieceBPETokenizer()
    sp_tokenizer.train([str(saving_path/"all_train_sentences.csv")],vocab_size=vocab_size, special_tokens = ["<unk>", "[CLS]", "[SEP]", "<s>", "</s>", "<pad>",">>de<<", ">>nds<<"])

    sp_tokenizer.save_model("model/transformers/","de_nds_tokenizer")





#%%
local_data = True

timestr = time.strftime("%Y%m%d-%H%M%S")
base_path = Path("model/transformers/")
saving_path =  base_path / timestr / "marianmt/"

if local_data == False:
    from google.colab import drive
    drive.mount('/content/drive')
    colab_path = Path("/content/drive/My Drive/capstone/colab_data")
    saving_path = colab_path / saving_path


# saving the model and results
try:
    # Create target Directory
    os.makedirs(saving_path)
    print("Directory " , saving_path ,  " Created ") 
except FileExistsError:
    print("Directory " , saving_path ,  " already exists") 




if local_data:
    path = Path("data_selection/20200723-165133/round_/8/")
    path = Path("preprocessed_data/")
    path = Path("preprocessed_data/tatoeba/tatoeba_dataset_cleaned_spelling.csv")
    path_round = path / "round_"
    train_path = path / "train_data.csv"
    valid_path = path / "valid_data.csv"
    test_path = path  / "test_data.csv"
else:
    # your link here
    path = colab_path
    train = pd.read_csv(path / "train_data.csv", index_col = 0)
    valid = pd.read_csv(path / "valid_data.csv", index_col = 0)
    wordbook = pd.read_csv(path / "wordbook.csv", index_col = 0)
    wordbook = wordbook[["deu", "nds"]]
    train = train.append(wordbook).sample(frac=1).reset_index(drop=True)
    mono = pd.read_csv(path / "train_mono_data.csv", index_col=0)
    print(mono.head(2))




#%%

model_name = 'Helsinki-NLP/opus-mt-NORTH_EU-NORTH_EU'

if model_name == 'Helsinki-NLP/opus-mt-NORTH_EU-NORTH_EU':
    custom_model_name = base_path / "mynewmodel_epoch_1"
    model = MarianMTModel.from_pretrained(str(custom_model_name))
    if custom_tokenizer == False:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        print(tokenizer.supported_language_codes)
    else:
        tokenizer = T5Tokenizer("model/transformers/shared_voc.model")
        model.base_model.resize_token_embeddings(vocab_size)
    prefix_deu = ">>nl<< "
    prefix_nds = ">>de<< "

else:
    #tok = T5Tokenizer("model/transformers/shared_voc.model")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    #model = T5ForConditionalGeneration.from_pretrained("t5-small", return_dict=True)
    model = T5ForConditionalGeneration.from_pretrained("model/transformers/20201104-161843/marianmt/mynewmodel", return_dict=True)
    prefix = "translate German to Low German: "
    prefix_trg = ""


model.to(device)

freeze_encoder = False
if freeze_encoder:
    for param in model.base_model.encoder.parameters():
        param.requires_grad = False

#%%


#%%
train.rename(columns={"deu": "src","nds":"trg"}, inplace=True)
valid.rename(columns={"deu": "src","nds":"trg"}, inplace=True)

train_reverse = train.copy()
valid_reverse = valid.copy()

train_reverse.rename(columns={"trg": "src","src":"trg"}, inplace=True)
valid_reverse.rename(columns={"trg": "src","src":"trg"}, inplace=True)
mono.rename(columns={"deu": "src","nds":"trg"}, inplace=True)

train["src"] = prefix_deu + train["src"]
valid["src"] = prefix_deu + valid["src"]

train_reverse["src"] = prefix_nds + train_reverse["src"]
valid_reverse["src"] = prefix_nds + valid_reverse["src"]

mono["src"] = prefix_deu + mono["src"]

train_all = train.append(train_reverse, ignore_index=True).append(mono,ignore_index=True)

valid_all = valid.append(valid_reverse, ignore_index = True)


train_dataset = tokenizer.prepare_seq2seq_batch(train_all["src"].tolist(), train_all["trg"].tolist(), return_tensors="pt",)
valid_dataset = tokenizer.prepare_seq2seq_batch(valid_all["src"].tolist(), valid_all["trg"].tolist())



#%%

train_all.head(2)

#%%


train_all.tail(2)



train_dataset.values()
#%%


for key, val in valid_dataset.items():
    print(key)
valid.applymap(len).max()

max([len(x) for x in valid_dataset["input_ids"]])
max([len(x) for x in valid_dataset["labels"]])



#%% 


#%%

train_data = NDSDataset(train_dataset)
train_loader = DataLoader(train_data, batch_size=11, shuffle=True, drop_last=True)

valid_data = NDSDataset(valid_dataset)
valid_loader = DataLoader(valid_data, batch_size=11, shuffle=True, drop_last=True)


print(len(train_loader))


#%%
import time
optim = AdamW(model.parameters(), lr=5e-5)

best_valid_loss = float('inf')


for epoch in range(6):
    model.train()
    time_col = 0
    #training
    print(f"------Epoch: {epoch} --------")
    for idx,batch in enumerate(train_loader):
        start = time.time()
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()
        end = time.time()
        time_col += end - start
        if idx % 100 == 0:
            print(idx)
            print(f"time {(time_col) / 60 } min")

    model.eval()
    valid_loss = 0
    # evaluation set
    for idx,batch in enumerate(valid_loader):
        start = time.time()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        loss = model(input_ids, attention_mask=attention_mask, labels=labels)[0].item()
        valid_loss += loss
    valid_loss = valid_loss / len(valid_loader)
    print(f"valid loss in epoch {epoch}: {valid_loss}")
    if valid_loss < best_valid_loss:
        model.save_pretrained(saving_path / "mynewmodel")
        print("new best model")
        best_valid_loss = valid_loss

    


#%%
test_df = pd.read_csv(path / "preprocessed_test_data.csv", index_col = 0)
test_df["deu"] = prefix_deu + test_df["deu"]
test_dataset = tokenizer.prepare_seq2seq_batch(test_df["deu"].tolist(),test_df["nds"].tolist())
test_data = NDSDataset(test_dataset)
test_loader = DataLoader(test_data, batch_size=8, shuffle=True, drop_last=True)
model.eval()
test_loss = 0
# evaluation set
for idx,batch in enumerate(test_loader):
    start = time.time()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    loss = model(input_ids, attention_mask=attention_mask, labels=labels)[0].item()
    test_loss += loss
test_loss = test_loss / len(test_loader)
print(f"test loss: {test_loss}")
#%%
def translate(text, trg_lang):
  input_ids = tokenizer(f"{trg_lang} {text}", return_tensors="pt").input_ids

  output_ids = model.generate(input_ids.to(device))

  return  tokenizer.decode(output_ids[0],skip_special_tokens=True)

#%%
text = "Nach dem Untergang eines Schlauchboots mit Flüchtlingen vor der libyschen Küste sind sechs Menschen ums Leben gekommen, darunter ein Baby. Rund hundert andere Insassen des völlig überfüllten Boots konnte die spanische Organisation Open Arms nach eigenen Angaben retten. Die Helfer mussten ins Wasser springen, um die Flüchtlinge zu bergen."
translate(text,">>nl<<")


 
 
 
 
#%%


text = "Dat is Leevke. Wullt du di ook even vöörstellen? För de, de Platt leren doot, wo wied köönt ji jo al vöörstellen?"
translate(text,">>de<<")


# %%


text = "Berlin: Wo de Corona-Tallen in Düütschland na baven gaht, maakt se mehr in mehr Scholen dicht. Dat lett: Opstunns sünd 300.000 Schoolkinner un bet 30.000 Schoolmesters in Karanteen. De Präsident vun den Düütschen Lehrerverband, Peter Meisinger, hett dat ARD-Hauptstadtstudio vertellt: Ja, wat dor in de Bild-Zeitung steiht, dat is wohr. De Lehrerverband meent: Gegen Corona warrt nich noog maakt. So wünscht he sik Ünnerricht in halve Gruppen."
translate(text,">>de<<")



# %%
