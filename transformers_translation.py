# %%


from transformers import MarianMTModel, MarianTokenizer, T5Tokenizer, T5ForConditionalGeneration
from pathlib import Path
import sentencepiece as spm
import pandas as pd
import time
import torch
from torchtext.data import Field, BucketIterator,  TabularDataset

# model_name = 'Helsinki-NLP/opus-mt-NORTH_EU-NORTH_EU'
# tokenizer = MarianTokenizer.from_pretrained(model_name)
# print(tokenizer.supported_language_codes)



# model = MarianMTModel.from_pretrained(model_name)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#%%
saving_path = Path("model/sentencepiece/20200728-152255/english_pretraining_with_sp")

#%%
# sp_nds = spm.SentencePieceProcessor(model_file=str(saving_path / "spm_nds.model"))
# sp_de = spm.SentencePieceProcessor(model_file=str(saving_path / "spm_deu.model"))
# #nds_tokenizer = MarianTokenizer([saving_path / "spm_nds.vocab", saving_path / "spm_deu.vocab"],saving_path / "spm_nds.model", saving_path / "spm_de.model")
# #tokenizer.spm_target = sp_nds

# sp_training_data = pd.read_csv(saving_path/"deu_train_sentences.csv", header=None).append(pd.read_csv(saving_path/"nds_train_sentences.csv",header=None))
# sp_training_data.to_csv(saving_path/"all_train_sentences.csv",index=None,header=None)
# sp_training_data
# #%%
# vocab_size = 20000
# sp_args = ''.join(["--control_symbols=>>de<<,>>nds<<",' --input=', str(saving_path/"all_train_sentences.csv")," --model_prefix=model/transformers/shared_voc", " --vocab_size=",str(vocab_size)])
# spm.SentencePieceTrainer.train(sp_args)

#%%
#sp_model = spm.SentencePieceProcessor(model_file="model/transformers/shared_voc.model")

#%%
from tokenizers import SentencePieceBPETokenizer

sp_tokenizer = SentencePieceBPETokenizer()
sp_tokenizer.train([str(saving_path/"all_train_sentences.csv")],vocab_size=20000, special_tokens = ["<unk>", "[CLS]", "[SEP]", "<s>", "</s>", "<pad>",">>de<<", ">>nds<<"])

sp_tokenizer.save_model("model/transformers/","de_nds_tokenizer")

#%%


#%%



#mt_tokenizer = MarianTokenizer("model/transformers/shared_voc.vocab","model/transformers/shared_voc.model","model/transformers/shared_voc.model")

#%%
src_text = [
    '>>sv<< Dieser Satz wird schwedisch.',
    '>>da<< Dieser Satz wird schwedisch.',
    '>>de<< Dieser Satz wird schwedisch.'
]
#%%
#tt = mt_tokenizer.encode("Moin Moin, ich gehe gleich nach Hause")
#mt_tokenizer.prepare_seq2seq_batch(["Hallo","Hier folgt jetzt ein sehr langer Satz."], ["Ja moin all tohoop"])


#%%

# translated = model.generate(**mt_tokenizer.prepare_seq2seq_batch(src_text))
# tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

# tgt_text
# %%

src_text = [
    '>>sv<< Dieser Satz wird schwedisch.'
]
nds_text = [
    '>>da<< Dieser Satz wird schwedisch.'
]



#%%

timestr = time.strftime("%Y%m%d-%H%M%S")
base_path = Path("model/transformers/")
saving_path =  base_path / timestr / "marianmt/"
# saving the model and results
try:
    # Create target Directory
    os.makedirs(saving_path)
    print("Directory " , saving_path ,  " Created ") 
except FileExistsError:
    print("Directory " , saving_path ,  " already exists") 



local_data = True

if local_data:
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
# loading in the data into torchtext datasets


#%%
["Hallo"]

#%%
from transformers import AdamW
from torch.utils.data import DataLoader
tok = T5Tokenizer("model/transformers/shared_voc.model")

train_dataset = tok.prepare_seq2seq_batch(train["deu"].tolist(), train["nds"].tolist(), return_tensors="pt",)

#%%


#%%
# from transformers import DataCollatorForLanguageModeling

# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tok, mlm=True, mlm_probability=0.15
# )

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


train_data = NDSDataset(train_dataset)

#%%

len(train_dataset["input_ids"])

#%%
train_loader = DataLoader(train_data, batch_size=20, shuffle=True, drop_last=True)

#%%
print(len(train_loader))

len(train_dataset)


#%%


model = T5ForConditionalGeneration.from_pretrained("t5-small", return_dict=True)
model.to(device)

#%%
import time
optim = AdamW(model.parameters(), lr=5e-5)
model.train()
for epoch in range(4):
    time_col = 0
    for idx,batch in enumerate(train_loader):
        start = time.time()
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        #outputs = model(tokenizer.prepare_seq2seq_batch(src_text, nds_text)
        loss = outputs[0]
        loss.backward()
        optim.step()
        end = time.time()
        time_col += end - start
        if idx % 100 == 0:
            print(idx)
            print(f"time {(time_col) / 60 } min")

    model.save_pretrained(saving_path / "mynewmodel.pt")


model.eval()


#%%



#%%
input_ids = tok("Hallo, na das klappt ja noch nicht so gut.", return_tensors="pt").input_ids

output_ids = model.generate(input_ids)

tok.decode(output_ids[0])


#%%




from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir="./NDS",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    #data_collator=data_collator,
    train_dataset=train_iterator,
    eval_dataset=valid_iterator,
    prediction_loss_only=True,
)

for param in model.base_model.encoder.parameters():
    param.requires_grad = False

model.to(device)


# %%
#trainer.train()
train_iterator
# %%
#labels = torch.tensor([1,0]).unsqueeze(0)
outputs = model(train_iterator)
loss = outputs.loss
loss.backward()
optimizer.step()
# %%
