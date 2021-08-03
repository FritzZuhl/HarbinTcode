#!/usr/bin/env python
# coding: utf-8

# # Text Classification with Word Embeddings and Dense Neural Network Models
# 
# Understanding the text content and predicting the sentiment of the reviews is a form of supervised machine learning. To be more specific, we will be using classification models for solving this problem. We will be building an automated sentiment text classification system in subsequent sections. The major steps to achieve this are mentioned as follows.
# 
# + Prepare train and test datasets (optionally a validation dataset)
# + Pre-process and normalize text documents
# + Feature Engineering 
# + Model training
# + Model prediction and evaluation
# 
# These are the major steps for building our system. Optionally the last step would be to deploy the model in your server or on the cloud. The following figure shows a detailed workflow for building a standard text classification system with supervised learning (classification) models.
# 
# <img src="https://github.com/dipanjanS/nlp_workshop_dhs18/blob/master/Unit%2012%20-%20Project%209%20-%20Sentiment%20Analysis%20-%20Supervised%20Learning/sentiment_classifier_workflow.png?raw=1">
# 
# In our scenario, documents indicate the movie reviews and classes indicate the review sentiments which can either be positive or negative making it a binary classification problem. We will build models using deep learning in the subsequent sections.

# # New Section

# # New Section

# In[1]:


get_ipython().system('nvidia-smi')


# In[2]:


get_ipython().system('pip install contractions')
get_ipython().system('pip install textsearch')
get_ipython().system('pip install tqdm')
import nltk
nltk.download('punkt')


# ## Load Dataset
# 
# Let's load our movie review dataset containing about 50000 reviews and their corresponding sentiments like positive and negative

# In[3]:


import pandas as pd
from google.colab import drive
drive.mount('/content/drive')


# In[4]:


print(pd.__version__)


# In[5]:



dataset = pd.read_csv('/content/drive/My Drive/NLP_DeepLearning_Course/Week1/movie_reviews.csv.bz2')
dataset.info()
frac = 1


# In[6]:


# downsample if needed
frac = 0.2
dataset = dataset.sample(frac=frac, random_state=253)
dataset.info()


# In[6]:





# In[7]:


dataset.head()


# ## Split Dataset into Train and Test sets
# 
# Since sentiment analysis is a supervised learning task, we split our movie review dataset into train and test sets

# In[8]:


# build train and test datasets
reviews = dataset['review'].values
sentiments = dataset['sentiment'].values

train_reviews = reviews[:int(35000*frac)]
train_sentiments = sentiments[:int(35000*frac)]

test_reviews = reviews[int(35000*frac):]
test_sentiments = sentiments[int(35000*frac):]


# In[9]:


print(train_reviews.shape)
print(train_sentiments.shape)
print(test_reviews.shape)
print(test_sentiments.shape)


# ## Text Wrangling and Normalization
# 
# The movie reviews have been collected by scraping web content. Typically scrapped data contains HTML tags and other pieces of information which can be easily discarded.
# 
# In this section, we will also normalize our corpus by removing accented characters, newline characters and so on. Lets get started

# In[10]:


import contractions
from bs4 import BeautifulSoup
import numpy as np
import re
from tqdm import tqdm
import unicodedata


def strip_html_tags(text):
  soup = BeautifulSoup(text, "html.parser")
  [s.extract() for s in soup(['iframe', 'script'])]
  stripped_text = soup.get_text()
  stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
  return stripped_text

def remove_accented_chars(text):
  text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
  return text

def pre_process_corpus(docs):
  norm_docs = []
  for doc in tqdm(docs):
    # strip HTML tags
    doc = strip_html_tags(doc)
    # remove extra newlines
    doc = doc.translate(doc.maketrans("\n\t\r", "   "))
    # lower case
    doc = doc.lower()
    # remove accented characters
    doc = remove_accented_chars(doc)
    # fix contractions
    doc = contractions.fix(doc)
    # remove special characters\whitespaces
    # use regex to keep only letters, numbers and spaces
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, flags=re.I|re.A)
    # use regex to remove extra spaces
    doc = re.sub(' +', ' ', doc)
    # remove trailing and leading spaces
    doc = doc.strip()  
    norm_docs.append(doc)
  
  return norm_docs


# In[11]:


get_ipython().run_cell_magic('time', '', '\nnorm_train_reviews = pre_process_corpus(train_reviews)\nnorm_test_reviews = pre_process_corpus(test_reviews)')


# ## Label Encode Class Labels
# 
# Our dataset has labels in the form of positive and negative classes. We transform them into consumable form by performing label encoding. Label encoding assigns a unique numerical value to each class. For example: 
# ``negative: 0 and positive:1``

# In[12]:


import gensim
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Activation, Dense
from sklearn.preprocessing import LabelEncoder


# In[13]:


le = LabelEncoder()
# tokenize train reviews & encode train labels
tokenized_train = [nltk.word_tokenize(text)
                       for text in tqdm(norm_train_reviews)]
y_train = le.fit_transform(train_sentiments)
# tokenize test reviews & encode test labels
tokenized_test = [nltk.word_tokenize(text)
                       for text in tqdm(norm_test_reviews)]
y_test = le.transform(test_sentiments)


# In[14]:


# print class label encoding map and encoded labels
print('Sentiment class label map:', dict(zip(le.classes_, le.transform(le.classes_))))
print('Sample test label transformation:\n'+'-'*35,
      '\nActual Labels:', test_sentiments[:33], '\nEncoded Labels:', y_test[:33])


# ## Feature Engineering based on Word2Vec Embeddings
# 
# In the previous notebook we discussed different word embedding techniques like word2vec, glove, fastText, etc. In this section we will leverage ``gensim`` to transform our dataset into word2vec  representation

# In[15]:


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[16]:


get_ipython().run_cell_magic('time', '', '# build word2vec model\nw2v_num_features = 300\nw2v_model = gensim.models.Word2Vec(tokenized_train, size=w2v_num_features, window=150,\n                                   min_count=10, workers=4, iter=5)    ')


# ## Feature Engineering based on FastText Embeddings
# 
# Similar to previous section, here will transform our corpus into FastText vectors using ``gensim``

# In[17]:


from gensim.models.fasttext import FastText

# Set values for various parameters
feature_size = 300    # Word vector dimensionality  
window_context = 50  # Context window size                                                                                    
min_word_count = 10   # Minimum word count                        
sample = 1e-3        # Downsample setting for frequent words
sg = 1               # skip-gram model

ft_model = FastText(tokenized_train, size=feature_size, 
                     window=window_context, min_count = min_word_count,
                     sg=sg, sample=sample, iter=2, workers=4)
ft_model


# ## Averaged Document Vectors
# 
# A sentence in very simple terms is a collection of words. By now we know how to transform words into vector representation. But how do we transform sentences and documents into vector representation?
# 
# A simple and naïve way is to average all words in a given sentence to form a sentence vector. In this section, we will leverage this technique itself to prepare our sentence/document vectors

# In[18]:


def averaged_doc_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)
    
    def average_word_vectors(words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype="float64")
        nwords = 0.
        
        for word in tqdm(words):
            if word in vocabulary: 
                nwords = nwords + 1.
                feature_vector = np.add(feature_vector, model.wv[word])
        if nwords:
            feature_vector = np.divide(feature_vector, nwords)

        return feature_vector

    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                    for tokenized_sentence in tqdm(corpus)]
    return np.array(features)


# In[19]:


# generate averaged word vector features from word2vec model
avg_w2v_train_features = averaged_doc_vectorizer(corpus=tokenized_train, model=w2v_model,
                                                     num_features=w2v_num_features)
avg_w2v_test_features = averaged_doc_vectorizer(corpus=tokenized_test, model=w2v_model,
                                                    num_features=w2v_num_features)


# In[20]:


print('Word2Vec model:> Train features shape:', avg_w2v_train_features.shape, 
      ' Test features shape:', avg_w2v_test_features.shape)


# In[21]:


# generate averaged word vector features from fastText model
avg_ft_train_features = averaged_doc_vectorizer(corpus=tokenized_train, model=ft_model,
                                                     num_features=feature_size)
avg_ft_test_features = averaged_doc_vectorizer(corpus=tokenized_test, model=ft_model,
                                                    num_features=feature_size)


# In[22]:


print('FastText model:> Train features shape:', avg_w2v_train_features.shape, 
      ' Test features shape:', avg_w2v_test_features.shape)


# ## Define DNN Model
# 
# Let us leverage ``tensorflow.keras`` to build our deep neural network for movie review classification task.
# We will make use of ``Dense`` layers with ``ReLU`` activation and ``Dropout`` to prevent overfitting.
# 
# Architecture used:
# 
# - 3 Dense Layers
# - 512 - 256 - 256 (neurons)
# - 20% dropout in each layer
# - 1 output layer for binary classification
# - binary crossentropy loss 
# - adam optimizer

# In[23]:


def construct_deepnn_architecture(num_input_features):
    dnn_model = Sequential()
    dnn_model.add(Dense(512, input_shape=(num_input_features,)))
    dnn_model.add(Activation('relu'))
    dnn_model.add(Dropout(0.2))
    
    dnn_model.add(Dense(256))
    dnn_model.add(Activation('relu'))
    dnn_model.add(Dropout(0.2))
    
    dnn_model.add(Dense(256))
    dnn_model.add(Activation('relu'))
    dnn_model.add(Dropout(0.2))
    
    dnn_model.add(Dense(1))
    dnn_model.add(Activation('sigmoid'))

    dnn_model.compile(loss='binary_crossentropy', optimizer='adam',                 
                      metrics=['accuracy'])
    return dnn_model


# ## Compile and Visualize Model

# In[24]:


w2v_dnn = construct_deepnn_architecture(num_input_features=w2v_num_features)


# In[25]:


w2v_dnn.summary()


# ## Train the Model using Word2Vec Features
# 
# The first exercise is to leverage word2vec features as input to our deep neural network to perform moview review classification

# In[26]:


batch_size = 100
w2v_dnn.fit(avg_w2v_train_features, y_train, epochs=10, batch_size=batch_size, 
            shuffle=True, validation_split=0.1, verbose=1)


# ### Evaluate Model

# In[27]:


from sklearn.metrics import confusion_matrix, classification_report


# In[28]:


y_pred = w2v_dnn.predict_classes(avg_w2v_test_features)
predictions = le.inverse_transform(y_pred) 


# In[29]:


labels = ['negative', 'positive']
print(classification_report(test_sentiments, predictions))
pd.DataFrame(confusion_matrix(test_sentiments, predictions), index=labels, columns=labels)


# The model seems to perform very nicely for both classes within a few iterations itself.

# ## Train the model using FastText Features
# 
# The second exercise we will perform using FastText feature vectors. Remember that we will use the same model architecture for this exercise as well but create a new instance of the same. Lets get started

# In[30]:


ft_dnn = construct_deepnn_architecture(num_input_features=feature_size)


# In[31]:


batch_size = 100
ft_dnn.fit(avg_ft_train_features, y_train, epochs=15, batch_size=batch_size, 
            shuffle=True, validation_split=0.1, verbose=1)


# ### Evaluate the model

# # New Section

# In[32]:


y_pred = ft_dnn.predict_classes(avg_ft_test_features)
predictions = le.inverse_transform(y_pred) 


# In[33]:


labels = le.classes_.tolist()
print(classification_report(test_sentiments, predictions))
pd.DataFrame(confusion_matrix(test_sentiments, predictions), index=labels, columns=labels)


# Amazing, FastText seems to identify both classes with a more balanced number of prediction errors than the model word2vec features. We encourage you to try out these models on other datasets too!
