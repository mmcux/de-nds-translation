#%%
import pandas as pd
from sklearn.model_selection import train_test_split

#%%

tatoeba_df = pd.read_csv("preprocessed_data/tatoeba/tatoeba_dataset_cleaned_spelling.csv", index_col = 0)
wiki_df = pd.read_csv("preprocessed_data/fb-wiki/wiki_dataset_cleaned_spelling.csv", index_col = 0)

print(len(wiki_df))
#wiki_raw = pd.read_csv("preprocessed_data/fb-wiki/train_wiki_data.csv", index_col = 0)

#%%

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
# %%

wrong_translation = wiki_df[wiki_df.nds.str.contains("De Sitt vun de Verw")]

print("De Sitt vun de Verwaltung sentences: ",len(wrong_translation))

wiki_df.drop(wrong_translation.index, inplace=True)
# %%
def get_sentence_range(df=wiki_df, start=wiki_df.index[0],end=70000):
    return df.loc[start:end,:]

#get_sentence_range(wiki_df,wiki_df.index[0],70000)


def get_train_valid_data(above_border=70000):
    wiki_selection = get_sentence_range(end=above_border)
    all_data = tatoeba_df.append(wiki_selection)
    train, valid = train_test_split(all_data, test_size = 0.1, random_state=35)
    return train, valid


# %%
train, valid = get_train_valid_data()
# %%
train.to_csv("preprocessed_data/train_data.csv")
valid.to_csv("preprocessed_data/valid_data.csv")


# %%
train_9 = pd.read_csv("data_selection/transformers_selection/new_training_data_9.csv")
train_10 = pd.read_csv("data_selection/transformers_selection/new_training_data_10.csv")


# %%
print( len(train_9), len(train_10))
# %%
mono = pd.read_csv("preprocessed_data/monolingual.csv", index_col = 0)

mono
# %%
