# de-nds-translation

Neural Machine Translation from German to the low-resource language Low German.
------

This repo contains the workflow for creating the first specialized "German" - "Low German" translator [översetter.de](http://www.översetter.de/).

If you have suggestions for improvement and/or want to participate in another way, please don't hesitate to get in contact with me.

---

# Low German

Low German is a language spoken in Northern Germany. Once the dominant language, it disappeared nowadays in daily usage and if spoken then mostly in old generations. It is listed as [endangered language](http://www.unesco.org/new/en/culture/themes/endangered-languages/atlas-of-languages-in-danger/).

Low German has its own vocabulary and grammar which prevents a word by word translation from German to Low German.

With the help of Neural Machine Translation this project wants to support the community which is working every day to keep the language alive.

<img src="https://unsplash.com/photos/T1IrtN3g8i8/download?force=true&w=640" alt="Low German" width="150"/>
Photo by [Marian on Unsplash](https://unsplash.com/@minjax)


---

# Low German as low resource language

Although the total speaker number with 1-2 million speaker might be relatively high compared to other low resource languages, [99.2% of the people under 20](http://www.ins-bremen.de/fileadmin/ins-bremen/user_upload/umfrage2016/broschuere-umfrage.pdf) can't speak Low German in Northern Germany.
As it is mainly present in very old generations, also the online resources of Low German are limited.

## Available data

If you have any Low German data which could be used for improving the translations, please let me know!

Beside that I have found two datasets, [Tatoeba](https://tatoeba.org/eng/) and [WikiMatrix](https://github.com/facebookresearch/LASER/tree/master/tasks/WikiMatrix), which fulfil two aspects, which were crucial for starting: Digital available and aligned sentences in German and Low German like this one:

|    German                              |    Low German                    |
| ---------------------------------------|----------------------------------|
|    Er hat mich lange warten lassen.    |    He hett mi lang töven laten.  |
|    Sie wollen reich werden.            |    Se wüllt riek warrn.          |
|    Niemand hat diesen Satz gelöscht.   |    Nüms hett dissen Satz wegdaan.|



In the notebook "data_preprocessing.ipynb" you can see how the datasets where preprocessed.


## Selecting right sentences

As there are many misalignments especially in the WikiMatrix dataset, you will find in self_learning_model.py an algorithm which selects the best sentences from the dataset.

This is the main work in working with this low resource language. Here you will see that the algorithm performs better than a baseline model which sees exactly the same data and could be used for any other language where the WikiMatrix dataset contains many mismatches. The neural network itself is constructed by [Ben Trevett](https://github.com/bentrevett/pytorch-seq2seq) and adapted for our purpose.

We train a model on the basis on the Tatoeba dataset where we can be sure that the sentences contain the same conten. We select than from a subset of the WikiMatrix data the best sentences (the ones with the lowest loss). In the next round the training data contains these sentences and looks at another subset of WikiMatrix data and selects again the best sentences from this subset.

![Self-selecting model](https://llmm.webo.family/index.php/s/ncTiyxP8mW9DB7s/download)




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
  * Batch size to 128
  * Learning Rate
  * Dropout
* better word correction / automatic input correction

## Dataset

* more data

## App

* user interface and logo
  * create possibility to flag / correct false translations
* create possibility for translating already existing sentences by community (more possible translations)
* get web address (översetter.de / oeversetter.de / ...)
* mobile friendly




```python

```
