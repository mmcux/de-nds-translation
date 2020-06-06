# de-nds-translation

Neural Machine Translation from German to the low-resource language Low German.
------

This repo contains the workflow for creating the first specialized "German" - "Low German" translator [översetter.de](http://www.översetter.de/).
<br>
<img src="https://llmm.webo.family/index.php/s/WBMKp4oBEsKSSeo/download" alt="Översetter" width="250"/>
<br>
If you have suggestions for improvement and/or want to participate in another way, please don't hesitate to get in contact with me.

---
# Quick Workflow-Overview

1. Gathering and preprocessing data: [data_preprocessing.ipynb](https://github.com/mmcux/de-nds-translation/blob/master/data_preprocessing.ipynb)
2. Data selection: [self_learning_model.py](https://github.com/mmcux/de-nds-translation/blob/master/self_learning_model.py)
3. Final model for translating: [translation_model.py](https://github.com/mmcux/de-nds-translation/blob/master/translation_model.py)

<img src="https://llmm.webo.family/index.php/s/c2sDqJCoR664bFr/download" alt="workflow" width="480"/>

Minor parts of the projects can't be published right now due to only personal permission for using the data. Still, you should be able to run all with the uploaded content.

---

# Low German

Low German is a language spoken in Northern Germany. Once the dominant language until the mid of 20th century, it disappeared nowadays in daily usage. Although statistics count 1-2 million Low German speakers, which might be relatively high compared to other low resource languages, [99.2% of the people under 20](http://www.ins-bremen.de/fileadmin/ins-bremen/user_upload/umfrage2016/broschuere-umfrage.pdf) can't speak Low German in Northern Germany. The language is dying rapidly and therefore it is listed as [endangered language by the UNESCO](http://www.unesco.org/new/en/culture/themes/endangered-languages/atlas-of-languages-in-danger/).

With the help of Neural Machine Translation this project wants to support the community which is working every day to keep the language alive.

<img src="https://unsplash.com/photos/T1IrtN3g8i8/download?force=true&w=640" alt="Low German" width="150"/>
Photo by [Marian on Unsplash](https://unsplash.com/@minjax)


---

# Low German as low resource language


As it is mainly present in very old generations, also the online resources of Low German are limited. Moreover Low German has its own vocabulary and grammar which prevents a word by word translation from German to Low German.
Another characteristic of Low German is the wide variety of spelling. In each region you have a slightly different Low German with its own dialect and its own spelling. For the same Low German word you might find several writings. The online dictionairy from [Peter Hansen](http://niederdeutsche-literatur.de/) gives a good overview about the different spellings.


## Available data

If you have any Low German data which could be used for improving the translations, please let me know!

Beside that I have found two datasets, [Tatoeba](https://tatoeba.org/eng/) and [WikiMatrix](https://github.com/facebookresearch/LASER/tree/master/tasks/WikiMatrix), which fulfil two aspects, which were crucial for starting: Digital available and aligned sentences in German and Low German like this one:

|    German                              |    Low German                    |
| ---------------------------------------|----------------------------------|
|    Er hat mich lange warten lassen.    |    He hett mi lang töven laten.  |
|    Sie wollen reich werden.            |    Se wüllt riek warrn.          |
|    Niemand hat diesen Satz gelöscht.   |    Nüms hett dissen Satz wegdaan.|



In the notebook "data_preprocessing.ipynb" you can see how the datasets where preprocessed.

You can download the Tatoeba tsv files from the [website](https://tatoeba.org/eng/downloads). The data is provided under the [CC BY 2.0 FR license](https://creativecommons.org/licenses/by/2.0/de/#).
Facebook aligned through all languages of Wikipedia suitable sentences and published it on: https://github.com/facebookresearch/LASER/tree/master/tasks/WikiMatrix
The corresponding paper was published from: Holger Schwenk, Vishrav Chaudhary, Shuo Sun, Hongyu Gong and Paco Guzman, [WikiMatrix: Mining 135M Parallel Sentences in 1620 Language Pairs from Wikipedia arXiv](https://arxiv.org/abs/1907.05791), July 11 2019. The data is provided under the Creative Commons Attribution-ShareAlike license.

As the sentences are written by volunteers from different regions, there is no common spelling within the sentences.
With the personal permission of Peter Hansen (thank you very much), I scraped his [online dictionairy](http://niederdeutsche-literatur.de/) with different possible spellings and replaced it by his proposal. With this method I could replace around 5% of the words to get a more uniform spelling. You will find an abstracted version of this in the [data_preprocessing.ipynb](https://github.com/mmcux/de-nds-translation/blob/master/data_preprocessing.ipynb).


## Selecting right sentences

The WikiMatrix dataset is built automatically and there was no quality control by a professional. The dataset includes misalignemnts even for sentences where Facebook Research was pretty sure that these sentences are aligned correctly. Therefore the main challenge for this dataset is to extract the right sentence pairs from the dataset.
You will find in self_learning_model.py an algorithm based on a Transformer-Seq2Seq model which selects the best sentences from the dataset.

The neural network basis construction is by [Ben Trevett](https://github.com/bentrevett/pytorch-seq2seq) and adapted for our purpose.

Main idea is to train a model on sentence pairs where we know that these are correct. After that we take sentence pairs with unknown quality (WikiMatrix dataset) and translate with our model the German sentence into an artificial Low German translation and calculate the error to the provided translation. Our model can already translate roughly, so the sentences with a low error probably contain somehow the same content even if the artifical translation is not correct. Or said the other way round: the sentences with a high error are significantly different to what the model learned before. This could have two reasons: The sentence pair could have the same content but on a higher level and our model is not good enough to translate this sentence OR we have a misalignment where the German and Low German sentence doesn't contain the same content.
Therefore we select the sentences with the lowest error and include them into our training set.



![Self-selecting model](https://llmm.webo.family/index.php/s/PSyQy22gSAoYJeK/download)

With this approach I was able to beat a random-pick baseline model which saw the same data fragments, but picked from these subsets random sentences and achieved better translation results. A more in depth analysis you will find in this [Google Presentation](https://docs.google.com/presentation/d/1-k97OhQeJvNb7LOp1YmvM_uFXjmrL8MxLcaKbGJlag4/edit?usp=sharing) or you just simply contact me for further questions.


# Future Work

To Do's for the future

## Algorithm

* Better tokenizer for nds
* Training with capitalized words
* monolingual training -> translation into german with wiki-tatoeba model -> backtranslation into low german
* Try model with more layers
* Use other pre-trained model: e.g. [OpenNMT](https://opennmt.net/Models-py/)
* Combine the output of multiple models (if possible)
  * Evaluation-Model: Trained model only on monolingual data to evaluate which predicted sentence is best
* k-fold cross-validation
* Hyperparameter tuning
  * Learning Rate
  * Dropout
* better word correction / automatic input correction

## Dataset

* more data

## App

* user interface and logo
* create possibility for translating already existing sentences by community (more possible translations)
* mobile friendly

# Licenses

The code is licensed under the MIT License.

The modified data (corrected spelling) will be uploaded soon under the [Creative Commons Attribution-ShareAlike license](https://creativecommons.org/licenses/by-sa/2.5/).


# Changelog

* 14.04.2020: Start of the project
* 10.05.2020: Publishing first prototype online
* 13.05.2020: Upper - & lowercase prediction together with model improvements
* 02.06.2020: Community feedback and correction function



