#!/usr/bin/env python
# coding: utf-8

# # Word Embeddings
# 
# A word embedding is a learned dense representation for text where words that have the same meaning have a similar representation. It is this approach to representing words and documents that may be considered one of the key breakthroughs of deep learning on challenging natural language processing problems.
# 
# In this notebook we will:
# + Understand the Word2Vec models called Skipgram and CBOW
# + Build our understanding towards GloVe Model
# + Learn about FastText and how it overcomes some of the limitations of Word2Vec and GloVe.
# 
# We will use libraries like ``gensim`` and ``spacy`` to go through some hands-on exercises as well.

# ## Exploring Word Embeddings with New Deep Learning Models
# 
# Traditional (count-based) feature engineering strategies for textual data involve models belonging to a family of models popularly known as the Bag of Words model. This includes term frequencies, TF-IDF (term frequency-inverse document frequency), N-grams and so on. While they are effective methods for extracting features from text, due to the inherent nature of the model being just a bag of unstructured words, we lose additional information like the semantics, structure, sequence and context around nearby words in each text document.
# 
# This forms as enough motivation for us to explore more sophisticated models which can capture this information and give us features which are vector representation of words, popularly known as embeddings.
# 
# Here we will explore the following feature engineering techniques:
# 
# + Word2Vec
# + GloVe
# + FastText
# 
# Predictive methods like Neural Network based language models try to predict words from its neighboring words looking at word sequences in the corpus and in the process it learns distributed representations giving us dense word embeddings. We will be focusing on these predictive methods in this article.

# ## Prepare a Sample Corpus
# 
# Let’s now take a sample corpus of documents on which we will run most of our analyses in this article. A corpus is typically a collection of text documents usually belonging to one or more subjects or domains.

# In[1]:


import pandas as pd
import numpy as np

pd.options.display.max_colwidth = 200

corpus = ['The sky is blue and beautiful.',
          'Love this blue and beautiful sky!',
          'The quick brown fox jumps over the lazy dog.',
          "A king's breakfast has sausages, ham, bacon, eggs, toast and beans",
          'I love green eggs, ham, sausages and bacon!',
          'The brown fox is quick and the blue dog is lazy!',
          'The sky is very blue and the sky is very beautiful today',
          'The dog is lazy but the brown fox is quick!'    
]
labels = ['weather', 'weather', 'animals', 'food', 'food', 'animals', 'weather', 'animals']

corpus = np.array(corpus)
corpus_df = pd.DataFrame({'Document': corpus, 
                          'Category': labels})
corpus_df = corpus_df[['Document', 'Category']]
corpus_df


# ## Simple Text Pre-processing
# Since the focus of this unit is on feature engineering, we will build a simple text pre-processor which focuses on removing special characters, extra whitespaces, digits, stopwords and lower casing the text corpus.

# In[2]:


import nltk
import re

nltk.download('stopwords')
nltk.download('punkt')

stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, flags=re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = nltk.word_tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)

norm_corpus = normalize_corpus(corpus)
norm_corpus


# ## The Word2Vec Model
# This model was created by Google in 2013 and is a predictive deep learning based model to compute and generate high quality, distributed and continuous dense vector representations of words, which capture contextual and semantic similarity. Essentially these are unsupervised models which can take in massive textual corpora, create a vocabulary of possible words and generate dense word embeddings for each word in the vector space representing that vocabulary.
# 
# Usually you can specify the size of the word embedding vectors and the total number of vectors are essentially the size of the vocabulary. This makes the dimensionality of this dense vector space much lower than the high-dimensional sparse vector space built using traditional Bag of Words models.
# 
# There are two different model architectures which can be leveraged by Word2Vec to create these word embedding representations. These include,
# 
# + The Continuous Bag of Words (CBOW) Model
# + The Skip-gram Model

# ### The Continuous Bag of Words (CBOW) Model
# The CBOW model architecture tries to predict the current target word (the center word) based on the source context words (surrounding words).
# 
# Considering a simple sentence, __“the quick brown fox jumps over the lazy dog”__, this can be pairs of __(context_window, target_word)__ where if we consider a context window of size 2, we have examples like __([quick, fox], brown), ([the, brown], quick), ([the, dog], lazy)__ and so on.
# 
# Thus the model tries to predict the target_word based on the context_window words.
# 
# <img src="https://i.imgur.com/ATyNx6u.png">

# ### The Skip-gram Model
# The Skip-gram model architecture usually tries to achieve the reverse of what the CBOW model does. It tries to predict the source context words (surrounding words) given a target word (the center word).
# 
# Considering our simple sentence from earlier, __“the quick brown fox jumps over the lazy dog”__. If we used the CBOW model, we get pairs of (context_window, target_word) where if we consider a context window of size 2, we have examples like __([quick, fox], brown), ([the, brown], quick), ([the, dog], lazy)__ and so on.
# 
# Now considering that the skip-gram model’s aim is to predict the context from the target word, the model typically inverts the contexts and targets, and tries to predict each context word from its target word. Hence the task becomes to predict the context __[quick, fox]__ given target word __‘brown’__ or __[the, brown]__ given target word __‘quick’__ and so on.
# 
# Thus the model tries to predict the context_window words based on the target_word.
# 
# <img src="https://i.imgur.com/95f3eVF.png">
# 
# Further details can be found in [Text Analytics with Python](https://github.com/dipanjanS/text-analytics-with-python/tree/master/New-Second-Edition/Ch04%20-%20Feature%20Engineering%20for%20Text%20Representation)

# ## Robust Word2Vec Model with Gensim
# The ``gensim`` framework, created by _Radim Řehůřek_ consists of a robust, efficient and scalable implementation of the Word2Vec model. We will leverage the same on our sample toy corpus. In our workflow, we will tokenize our normalized corpus and then focus on the following four parameters in the ``Word2Vec`` model to build it.
# 
# + ``size``: The word embedding dimensionality
# + ``window``: The context window size
# + ``min_count``: The minimum word count
# + ``sample``: The downsample setting for frequent words
# + ``sg``: Training model, 1 for skip-gram otherwise CBOW
# 
# We will build a simple Word2Vec model on the corpus and visualize the embeddings.

# In[3]:


tokenized_corpus = [nltk.word_tokenize(doc) for doc in norm_corpus]


# In[4]:


import nltk
from gensim.models import word2vec


# Set values for various parameters
feature_size = 15    # Word vector dimensionality  
window_context = 20  # Context window size                                                                                    
min_word_count = 1   # Minimum word count                        
sample = 1e-3        # Downsample setting for frequent words
sg = 1               # skip-gram model

w2v_model = word2vec.Word2Vec(tokenized_corpus, size=feature_size, 
                              window=window_context, min_count = min_word_count,
                              sg=sg, sample=sample, iter=5000)
w2v_model


# In[5]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

# visualize embeddings
from sklearn.manifold import TSNE

words = w2v_model.wv.index2word
wvs = w2v_model.wv[words]

tsne = TSNE(n_components=2, random_state=42, n_iter=5000, perplexity=4)
np.set_printoptions(suppress=True)
T = tsne.fit_transform(wvs)
labels = words

plt.figure(figsize=(12, 6))
plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
for label, x, y in zip(labels, T[:, 0], T[:, 1]):
    plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')


# In[6]:


# vector representation of a given word
w2v_model.wv['sky']


# In[7]:


# size of the vector
w2v_model.wv['sky'].shape


# In[8]:


# view sample vectors
vec_df = pd.DataFrame(wvs, index=words)
vec_df.head()


# ### Looking at term semantic similarity

# In[9]:


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(vec_df.values)
similarity_df = pd.DataFrame(similarity_matrix, index=words, columns=words)
similarity_df.head(10)


# In[10]:


feature_names = np.array(words)
similarity_df.apply(lambda row: feature_names[np.argsort(-row.values)[1:4]], 
                    axis=1)


# ## Glove Embeddings
# 
# The GloVe model stands for Global Vectors which is an unsupervised learning model which can be used to obtain dense word vectors similar to Word2Vec. However the technique is different and training is performed on an aggregated global word-word co-occurrence matrix, giving us a vector space with meaningful sub-structures. This method was invented in Stanford by Pennington et al. and I recommend you to read the original paper on GloVe, [‘GloVe: Global Vectors for Word Representation’ by Pennington et al](https://nlp.stanford.edu/pubs/glove.pdf). which is an excellent read to get some perspective on how this model works.
# 
# The basic methodology of the GloVe model is to first create a huge word-context co-occurence matrix consisting of (word, context) pairs such that each element in this matrix represents how often a word occurs with the context (which can be a sequence of words). The idea then is to apply matrix factorization to approximate this matrix as depicted in the following figure.
# 
# <img src="https://i.imgur.com/FnWASi2.png">
# 
# Considering the __Word-Context (WC)__ matrix, __Word-Feature (WF)__ matrix and __Feature-Context (FC)__ matrix, we try to factorize __$WC = WF x FC$__
# 
# Such that we we aim to reconstruct __WC__ from __WF__ and __FC__ by multiplying them. For this, we typically initialize __WF__ and __FC__ with some random weights and attempt to multiply them to get __WC'__ (an approximation of WC) and measure how close it is to WC. We do this multiple times using Stochastic Gradient Descent (SGD) to minimize the error. Finally, the __Word-Feature matrix (WF)__ gives us the word embeddings for each word where __F__ can be preset to a specific number of dimensions

# ## Robust Glove Model with SpaCy
# Let’s try and leverage GloVe based embeddings for our document clustering task. The very popular spacy framework comes with capabilities to leverage GloVe embeddings based on different language models. You can also get pre-trained word vectors and load them up as needed using gensim or spacy.
# 
# If you have spacy installed, we will be using the __[``en_vectors_web_lg model``](https://spacy.io/models/en#en_vectors_web_lg)__ which consists of 300-dimensional word vectors trained on [Common Crawl](https://commoncrawl.org/) with GloVe.
# 
# ### Install Instructions:
# ```python
# # Use the following command to install spaCy
# > pip install -U spacy
# OR
# > conda install -c conda-forge spacy
# ```

# If you want to implement GloVe from scratch do check out [this tutorial](http://www.foldl.me/2014/glove-python/) which can get a bit involving given you need to compute the matrices to get to the embeddings.

# In[11]:


# commented out line, FCZ, July 13
# !python -m spacy download en


# In[12]:


import spacy

spacy.load('en')


# In[13]:


# commented out line below on July 13. it takes a long time to run.
# !python -m spacy download en_vectors_web_lg


# __NOTE__: If you are on colab, next cell may require a restart of the runtime
# 
# ```shell
# Menu > Runtime > Restart runtime
# ```

# In[14]:


# restart runtime and rerun if this fails
import spacy

nlp = spacy.load('en_vectors_web_lg')
total_vectors = len(nlp.vocab.vectors)

print('Total word vectors:', total_vectors)


# This validates that everything is working and in order. Let’s get the GloVe embeddings for each of our words now in our toy corpus.

# In[15]:


unique_words = list(set([word for sublist in tokenized_corpus for word in sublist]))

word_glove_vectors = np.array([nlp(word).vector for word in unique_words])
vec_df = pd.DataFrame(word_glove_vectors, index=unique_words)
vec_df


# We can now use t-SNE to visualize these embeddings similar to what we did using our Word2Vec embeddings.

# In[16]:


# visualize embeddings
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42, n_iter=5000, perplexity=4)
np.set_printoptions(suppress=True)
T = tsne.fit_transform(word_glove_vectors)
labels = unique_words

plt.figure(figsize=(12, 6))
plt.scatter(T[:, 0], T[:, 1], c='red', edgecolors='r')
for label, x, y in zip(labels, T[:, 0], T[:, 1]):
    plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')


# ### Looking at term semantic similarity

# In[17]:


similarity_matrix = cosine_similarity(vec_df.values)
similarity_df = pd.DataFrame(similarity_matrix, index=unique_words, columns=unique_words)
similarity_df


# In[18]:


feature_names = np.array(unique_words)
similarity_df.apply(lambda row: feature_names[np.argsort(-row.values)[1:4]], 
                    axis=1)


# ## FastText Model
# 
# The FastText model was first introduced by Facebook in 2016 as an extension and supposedly improvement of the vanilla Word2Vec model. Based on the original paper titled ‘[Enriching Word Vectors with Subword Information](https://arxiv.org/pdf/1607.04606.pdf)’ by Mikolov et al. which is an excellent read to gain an in-depth understanding of how this model works. Overall, FastText is a framework for learning word representations and also performing robust, fast and accurate text classification. The framework is open-sourced by Facebook on [GitHub](https://github.com/facebookresearch/fastText) and claims to have the following.
# 
# + Recent state-of-the-art English word vectors.
# + Word vectors for 157 languages trained on Wikipedia and Crawl.
# + Models for language identification and various supervised tasks.
# Though I haven't implemented this model from scratch, based on the research paper, following is what I learnt about how the model works. In general, predictive models like the Word2Vec model typically considers each word as a distinct entity (e.g. where) and generates a dense embedding for the word. However this poses to be a serious limitation with languages having massive vocabularies and many rare words which may not occur a lot in different corpora.
# 
# The Word2Vec model typically ignores the morphological structure of each word and considers a word as a single entity. The FastText model __considers each word as a Bag of Character n-grams__. This is also called as a __subword model__ in the paper.
# 
# We add special boundary symbols __$<$__ and __$>$__ at the beginning and end of words. This enables us to distinguish prefixes and suffixes from other character sequences. We also include the word w itself in the set of its n-grams, to learn a representation for each word (in addition to its character n-grams).
# 
# Taking the word where and __n=3 (tri-grams)__ as an example, it will be represented by the __character n-grams: <wh, whe, her, ere, re>__ and the special sequence __< where >__ representing the whole word. Note that the sequence , corresponding to the word __< her >__ is different from the tri-gram __her__ from the word __where__.
# 
# In practice, the paper recommends in extracting all the n-grams for __n ≥ 3__ and __n ≤ 6__. This is a very simple approach, and different sets of n-grams could be considered, for example taking all prefixes and suffixes. We typically associate a vector representation (embedding) to each n-gram for a word.
# 
# Thus, we can represent a word by the sum of the vector representations of its n-grams or the average of the embedding of these n-grams. Thus, due to this effect of leveraging n-grams from individual words based on their characters, there is a higher chance for rare words to get a good representation since their character based n-grams should occur across other words of the corpus.

# ## Robust FastText Model with Gensim
# The ``gensim`` package has nice wrappers providing us interfaces to leverage the FastText model available under the ``gensim.models.fasttext`` module. Let’s apply this once again on our toy corpus.

# In[19]:


from gensim.models.fasttext import FastText

# Set values for various parameters
feature_size = 15    # Word vector dimensionality  
window_context = 20  # Context window size                                                                                    
min_word_count = 1   # Minimum word count                        
sample = 1e-3        # Downsample setting for frequent words
sg = 1               # skip-gram model

ft_model = FastText(tokenized_corpus, size=feature_size, 
                     window=window_context, min_count = min_word_count,
                     sg=sg, sample=sample, iter=5000)
ft_model


# In[20]:


# visualize embeddings
from sklearn.manifold import TSNE

words = ft_model.wv.index2word
wvs = ft_model.wv[words]

tsne = TSNE(n_components=2, random_state=42, n_iter=5000, perplexity=5)
np.set_printoptions(suppress=True)
T = tsne.fit_transform(wvs)
labels = words

plt.figure(figsize=(12, 6))
plt.scatter(T[:, 0], T[:, 1], c='green', edgecolors='k')
for label, x, y in zip(labels, T[:, 0], T[:, 1]):
    plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')


# ## Embedding Operations

# In[21]:


ft_model['sky'], ft_model['sky'].shape


# ### Similarity between two given words

# In[22]:


print(ft_model.wv.similarity(w1='ham', w2='sky'))
print(ft_model.wv.similarity(w1='ham', w2='sausages'))


# ### Find the Odd One Out

# In[23]:


st1 = "dog fox ham"
print('Odd one out for [',st1, ']:',  
      ft_model.wv.doesnt_match(st1.split()))

st2 = "bacon ham sky sausages"
print('Odd one out for [',st2, ']:', 
      ft_model.wv.doesnt_match(st2.split()))


# ## Getting document level embeddings
# 
# Now suppose we wanted to cluster the eight documents from our toy corpus, we would need to get the document level embeddings from each of the words present in each document. One strategy would be to average out the word embeddings for each word in a document. This is an extremely useful strategy and you can adopt the same for your own problems. Let’s apply this now on our corpus to get features for each document.

# In[24]:


def average_word_vectors(words, model, vocabulary, num_features):
    
    feature_vector = np.zeros((num_features,),dtype="float64")
    nwords = 0.
    
    for word in words:
        if word in vocabulary: 
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model[word])
    
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
        
    return feature_vector


def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                    for tokenized_sentence in corpus]
    return np.array(features)


# In[25]:


# get document level embeddings
ft_doc_features = averaged_word_vectorizer(corpus=tokenized_corpus, model=ft_model,
                                             num_features=feature_size)
pd.DataFrame(ft_doc_features)


# ### Application: Document clustering
# Now that we have our features for each document, let’s cluster these documents using the Affinity Propagation algorithm, which is a clustering algorithm based on the concept of “message passing” between data points and does not need the number of clusters as an explicit input which is often required by partition-based clustering algorithms.

# In[26]:


from sklearn.cluster import AffinityPropagation

ap = AffinityPropagation()
ap.fit(ft_doc_features)

cluster_labels = ap.labels_
cluster_labels = pd.DataFrame(cluster_labels, 
                              columns=['ClusterLabel'])

pd.concat([corpus_df, cluster_labels], axis=1)


# We can see that our algorithm has clustered each document into the right group based on our __Word2Vec__ features. Pretty neat! We can also visualize how each document in positioned in each cluster by using __Principal Component Analysis (PCA)__ to reduce the feature dimensions to 2-D and then visualizing the same (by color coding each cluster).

# In[27]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(ft_doc_features)
labels = ap.labels_
categories = list(corpus_df['Category'])
plt.figure(figsize=(8, 6))

for i in range(len(labels)):
    label = labels[i]
    color = 'orange' if label == 0 else 'blue' if label == 1 else 'green'
    annotation_label = categories[i]
    x, y = pcs[i]
    plt.scatter(x, y, c=color, edgecolors='k')
    plt.annotate(annotation_label, xy=(x+1e-2, y+1e-2), xytext=(0, 0), 
                 textcoords='offset points')


# In[28]:


print("Finished.")


# Everything looks to be in order as documents in each cluster are closer to each other and far apart from other clusters.
