#!/usr/bin/env python
# coding: utf-8

# # Deep Learning Essentials with ``TensorFlow``
# 
# Gain a high-level understanding of essentials of deep neural networks. Learn about some essential components of neural networks like layers, loss functions, activations, etc.
# 
# In this notebook, we will cover:
# 
# + Tensors and Operations
# + Building Deep Neural Network for a sample dataset to perform linear regression
# + Learn about Sequential, Functional and SubClass APIs from ``tensorflow.keras``
# + Learn about different activation functions
# + Learn about gradient descent and autodiff with Gradient Tape (backpropagation principles)
# + Build an DNN image classifier using tensorflow
# + Learn about Batch Normalization and Dropout Layers
# 

# In[1]:


import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf





print(tf.__version__)


# ## Tensors and Ops
# 
# Tensors are multi-dimensional arrays with a uniform type (called a ``dtype``). You can see all supported dtypes at ``tf.dtypes.DType``. If you're familiar with ``numpy``, tensors are (kind of) like ``np.arrays``.
# 
# All tensors are immutable like Python numbers and strings: you can never update the contents of a tensor, only create a new one.
# 
# 

# ### Constants as Tensors
# 
# The name ``tf.constant`` comes from the value being embeded in a _Const_ node in the ``tf.Graph``. ``tf.constant`` is useful for asserting that the value can be embedded that way.




t = tf.constant([[1., 2., 3.], 
                 [4., 5., 6.]])
t





# data type
t.dtype





# shape
t.shape


# ### Variables as Tensors
# 
# A TensorFlow variable is the recommended way to represent shared, persistent state your program manipulates. This guide covers how to create, update, and manage instances of ``tf.Variable`` in TensorFlow.
# 
# Variables are created and tracked via the ``tf.Variable`` class. A ``tf.Variable`` represents a tensor whose value can be changed by running ops on it. Specific ops allow you to read and modify the values of this tensor. Higher level libraries like ``tf.keras`` use ``tf.Variable`` to store model parameters.




t = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
t





# convert to numpy
t.numpy()





# assign value to tensor
t[0,0].assign(99)
t





# incorrect method
t[1] = [10, 20, 30]





# correct method
t[1].assign([10, 20, 30])


# ### Indexing
# 
# Indexing of tensors is similar to ``numpy``. The following examples showcase typical ways of indexing a tensor




t[1:, :]





t[1:, ...]





t[..., 1:]


# ### Basic Ops
# 
# TensorFlow tensors support all typical mathematical operations. They also support a number of inbuilt functions/utilities like square, transpose, etc.




# addition
t = t + 10
t





# inbuilt utils
tf.square(t)





# inbuilt utils
t @ tf.transpose(t)


# ## Linear Regression using ``sklearn``
# 
# Before we start developing deep neural networks, let us build a baseline using ``sklearn``.
# 
# We will use the sample __california housing__ dataset available as part of ``sklearn.datasets`` API itself. The datasets contains a number of attributes per listing (like Median Income, House Age, Average number of rooms, etc) and the aim is to predict the price of the house.
# 
# We will build a simple linear regression model for this given task




from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# this is needed to get data from sklearn.datasets (at least of the 'fetch' variety)
# do not know how this works. I consider myself smart enough at the moment.
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context




housing = fetch_california_housing()





X = pd.DataFrame(housing['data'], columns=housing['feature_names'])
X.head()





y = pd.DataFrame({'price': housing['target']})
y.head()





# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train.shape, X_test.shape





# scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)





# fit model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)





# evaluate model
predictions = lr.predict(X_test_scaled)
print('R2:', r2_score(y_test, predictions))
print('MSE:', mean_squared_error(y_test, predictions))


# ## TF Sequential API: Create a Simple 1-layer NN
# 
# A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
# 
# TensorFlow provides the high level API Sequential through ``tensorflow.keras`` to develop models. In this setting:
# 
# + We can pass different layers (``tf.keras.layers``) in the form of a list of the ``tf.keras.models.Sequential`` class.
# + We can also use the ``add()`` method of the instantiated object of the ``tf.keras.models.Sequential`` class
# 
# 
# 
# In this section, we will build a neural network based regression model to predict housing prices. We will reuse the train-test split from the ``sklearn`` exercise for consitentcy.




# define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation="relu", 
                          input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1),
])


# ### TF: Creating custom loss functions
# 
# TensorFlow provides a long list of popularly used loss functions such as __mean_squared_error__ but also provides an easy way to develop some of our own. Let us try to recreate mean squared error on our own here.
# 
# Create a ``mse_loss(...)`` function with two arguments:
# 
# + the true labels ``y_true``
# + the model predictions ``y_pred``
# 
# Make it return the mean squared error using TensorFlow operations. Note that you could write your own custom metrics in this way.
# 
# __Tip__: Recall that the MSE is the mean of the squares of prediction errors, which are the differences between the predictions and the labels, so you will need to use ``tf.reduce_mean()`` and ``tf.square()`` ops.




def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))





# compile the model
model.compile(loss=mse_loss, 
              optimizer=tf.keras.optimizers.SGD(lr=1e-3),
              metrics=['mean_squared_error'])





# get model summary
model.summary()





# train/fit the model
history = model.fit(X_train_scaled, y_train, 
                    epochs=30,
                    batch_size=32,
                    validation_split=0.1)





# visualize training progress
history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot(kind='line')





# evaluate model
predictions = model.predict(X_test_scaled)
print('R2:', r2_score(y_test, predictions))
print('MSE:', mean_squared_error(y_test, predictions))


# ## Activation Functions
# 
# Activation functions are mathematical equations that determine the output of a neural network. The function is attached to each neuron in the network, and determines whether it should be activated (“fired”) or not, based on whether each neuron’s input is relevant for the model’s prediction.
# 
# An additional aspect of activation functions is that they must be computationally efficient and differentiable. It can be as simple as a step function that turns the neuron output on and off, depending on a rule or threshold. Or it can be a transformation that maps the input signals into output signals that are needed for the neural network to function. The following are some of the widely used activation functions:
# 
# + Step
# + sigmoid
# + tanh
# + Rectified Linear Unit or ReLU
# 
# 
# Some of the recent ones are:
# + ELU
# + SELU
# + Swish
# + SoftPlus
# + LeakyReLU




z = np.linspace(-5, 5, 200)
plt.figure(figsize=(15,6))
plt.plot(z, np.sign(z), "r-", linewidth=1, label="Step")
plt.plot(z, tf.nn.sigmoid(z), "g--", linewidth=2, label="Sigmoid")
plt.plot(z, tf.nn.tanh(z), "b-", linewidth=2, label="Tanh")
plt.plot(z, tf.nn.relu(z), "m-.", linewidth=2, label="ReLU")
plt.grid(True)
plt.legend(loc="upper left", fontsize=14)
plt.title("Activation functions", fontsize=14)
plt.axis([-5, 5, -1.2, 2]);





# some more activation functions
z = np.linspace(-5, 5, 200)
plt.figure(figsize=(15,6))
plt.plot(z, np.sign(z), "r-", linewidth=1, label="Step")
plt.plot(z, tf.nn.elu(z), "k.", linewidth=2, label="ELU")
plt.plot(z, tf.nn.selu(z), "c.-", linewidth=2, label="SELU")
plt.plot(z, tf.nn.swish(z), "y--", linewidth=2, label="Swish")
plt.plot(z, tf.nn.softplus(z), "r--", linewidth=2, label="Softplus")
plt.plot(z, tf.nn.leaky_relu(z, alpha=0.05), "m--", linewidth=2, label="Leaky Relu")

plt.grid(True)
plt.legend(loc="upper left", fontsize=14)
plt.title("Activation functions", fontsize=14)
plt.axis([-5, 5, -1.2, 2])


# ## Layer Weight Initializers
# 
# Initializers define the way to set the initial random weights of TensorFlow layers.
# 
# The keyword arguments used for passing initializers to layers depends on the layer.
# 
# The following is a list of possible initializers available in the framework




[name for name in dir(tf.keras.initializers) if not name.startswith("_")]





# uniform
input_x = np.array([[1]])
dense1 = tf.keras.layers.Dense(100, activation="relu", kernel_initializer="uniform", input_shape=(input_x.shape[0],))
y = dense1(input_x)
plt.hist(dense1.weights[0][0]);





# glorot
input_x = np.array([[1]])
dense1 = tf.keras.layers.Dense(100, activation="relu", kernel_initializer="glorot_uniform", input_shape=(input_x.shape[0],))
y = dense1(input_x)
plt.hist(dense1.weights[0][0]);


# ## TF Functional API: 
# Not all neural network models are simply sequential. Some may have complex topologies. Some may have multiple inputs and/or multiple outputs. For example, a Wide & Deep neural network (see [paper](https://ai.google/research/pubs/pub45413)) connects all or part of the inputs directly to the output layer, as shown on the following diagram:
# 
# 
# ![](https://i.imgur.com/B6Y6coM.png)

# ### Build a Wide and Deep NN
# 
# Use ``keras'`` functional API to implement a Wide & Deep network to tackle the California housing problem.
# 
# __Tips__:
# 
# + You need to create a tf.keras.layers.Input layer to represent the inputs. + + Don't forget to specify the input shape.
# + Create the Dense layers, and connect them by using them like functions. For example, ``hidden1 = tf.keras.layers.Dense(30, activation="relu")(input)`` and ``hidden2 = tf.keras.layers.Dense(30, activation="relu")(hidden1)``
# + Use the ``tf.keras.layers.concatenate()`` function to concatenate the input layer and the previous hidden layer's output.
# + Create a tf.keras.models.Model and specify its inputs and outputs (e.g., ``inputs=[input]``).
# + Then use this model just like a Sequential model: you need to compile it, display its summary, train it, evaluate it and use it to make predictions.




# define the model
input = tf.keras.layers.Input(shape=(X_train.shape[1],))

x = tf.keras.layers.Dense(16, activation="relu")(input)
x = tf.keras.layers.Dense(32, activation="relu")(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)


concat = tf.keras.layers.concatenate([input, x])

output = tf.keras.layers.Dense(1)(concat)





# compile the model
model = tf.keras.models.Model(inputs=[input], outputs=[output])

model.compile(loss="mean_squared_error", 
              optimizer=tf.keras.optimizers.SGD(1e-3))
model.summary()





# visulize the model
tf.keras.utils.plot_model(model, show_shapes=True)





# train/fit the model
history = model.fit(X_train_scaled, y_train, 
                    epochs=30,
                    batch_size=32,
                    validation_split=0.1)





# plot training progress
history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot(kind='line')





# evaluate model
predictions = model.predict(X_test_scaled)
print('R2:', r2_score(y_test, predictions))
print('MSE:', mean_squared_error(y_test, predictions))


# ## TF SubClass API:
# 
# After the Sequential API and the Functional API, let's try the Subclassing API:
# 
# + Create a subclass of the tf.keras.models.Model class.
# + Create all the layers you need in the constructor (e.g., ``self.hidden1 = tf.keras.layers.Dense(...))``.
# + Use the layers to process the input in the ``call()`` method, and return the output.
# + Note that you do not need to create a ``tf.keras.layers.Input`` in this case.
# + Also note that ``self.output`` is used by Keras, so you should use another name for the output layer (e.g., ``self.output_layer``).
# 
# 
# __When should you use the Subclassing API?__
# 
# + Both the Sequential API and the Functional API are declarative: you first declare the list of layers you need and how they are connected, and only then can you feed your model with actual data. The models that these APIs build are just static graphs of layers. This has many advantages (easy inspection, debugging, saving, loading, sharing, etc.), and they cover the vast majority of use cases
# 
# + If you need to build a very dynamic model (e.g., with loops or conditional branching), or if you want to experiment with new ideas using an imperative programming style, then the Subclassing API is for you. You can pretty much do any computation you want in the call() method, possibly with loops and conditions, using Keras layers of even low-level TensorFlow operations.
# 
# + However, this extra flexibility comes at the cost of less transparency. Since the model is defined within the call() method, Keras cannot fully inspect it. All it sees is the list of model attributes (which include the layers you define in the constructor), so when you display the model summary you just see a list of unconnected layers. Consequently, you cannot save or load the model without writing extra code. So this API is best used only when you really need the extra flexibility.




# define a custom model class
class MyRegressionModel(tf.keras.models.Model):

    def __init__(self):
        super(MyRegressionModel, self).__init__()
        self.hidden1 = tf.keras.layers.Dense(16, activation="relu")
        self.hidden2 = tf.keras.layers.Dense(32, activation="relu")
        self.hidden3 = tf.keras.layers.Dense(32, activation="relu")
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, input):
        x = self.hidden1(input)
        x = self.hidden2(x)
        x = self.hidden3(x)
        concat = tf.keras.layers.concatenate([input, x])
        output = self.output_layer(concat)
        return output





# instantiate and compile custom model object
model = MyRegressionModel()
model.compile(loss="mean_squared_error", 
              optimizer=tf.keras.optimizers.SGD(1e-3))





# train/fit the model
history = model.fit(X_train_scaled, y_train, 
                    epochs=30,
                    batch_size=32,
                    validation_split=0.1)





# plot training progress
history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot(kind='line')





# evaluate model performance
predictions = model.predict(X_test_scaled)
print('R2:', r2_score(y_test, predictions))
print('MSE:', mean_squared_error(y_test, predictions))


# 
# 
# ---
# 
# 

# ## Gradient Descent & Autodiff with GradientTape

# ### Implement Gradient Descent manually 
# 
# Find the value of x that minimizes the following function f(x).
# 
# 




def f(x):
    return 5. * x ** 2 + 3. * x + 1.





f(1)





def approximate_diff(f, x, eps=1e-5):
    return (f(x + eps) - f(x - eps)) / (2. * eps)





approximate_diff(f, 1) # true derivative = 13


# ### Visualize function space




xs = np.linspace(-2, 2, 200)
fs = f(xs)
x0 = 0.25
df_x0 = approximate_diff(f, x0)
tangent_x0 = df_x0 * (xs - x0) + f(x0)
plt.plot([-2, 2], [0, 0], "k-", linewidth=1)
plt.plot([0, 0], [-5, 15], "k-", linewidth=1)
plt.plot(xs, fs)
plt.plot(xs, tangent_x0, "r--")
plt.plot(x0, f(x0), "ro")
plt.grid(True)
plt.xlabel("x", fontsize=14)
plt.ylabel("f(x)", fontsize=14, rotation=0)
plt.axis([-2, 2, -5, 15]);


# ### Gradient Tape for fast diff




x = tf.Variable(1.0)

with tf.GradientTape() as tape:
    z = f(x)
grads = tape.gradient(z, [x])
grads





x = tf.Variable(0.)

with tf.GradientTape() as tape:
    z = f(x)
grads = tape.gradient(z, [x])
grads


# ### Gradient descent with Gradient Tape




def f(x):
    return 5 * x ** 2 + 3 * x + 1.





xs = np.linspace(-2, 2, 200)
fs = f(xs)
x0 = 0.5
df_x0 = approximate_diff(f, x0)
tangent_x0 = df_x0 * (xs - x0) + f(x0)
plt.plot([-2, 2], [0, 0], "k-", linewidth=1)
plt.plot([0, 0], [-5, 15], "k-", linewidth=1)
plt.plot(xs, fs)
plt.plot(xs, tangent_x0, "r--")
plt.plot(x0, f(x0), "ro")
plt.grid(True)
plt.xlabel("x", fontsize=14)
plt.ylabel("f(x)", fontsize=14, rotation=0)
plt.axis([-2, 2, -5, 15]);





learning_rate = 0.1
x = tf.Variable(0.)

for i, epoch in enumerate(range(10)):
    with tf.GradientTape() as tape:
        z = f(x)
    dz_dx = tape.gradient(z, x)
    print('Epoch:', i, 'Grad:', dz_dx.numpy())
    x.assign_sub(learning_rate * dz_dx)





x.numpy()





f(x.numpy())





xs = np.linspace(-2, 2, 200)
fs = f(xs)
x0 = x.numpy()
df_x0 = approximate_diff(f, x0)
tangent_x0 = df_x0 * (xs - x0) + f(x0)
plt.plot([-2, 2], [0, 0], "k-", linewidth=1)
plt.plot([0, 0], [-5, 15], "k-", linewidth=1)
plt.plot(xs, fs)
plt.plot(xs, tangent_x0, "r--")
plt.plot(x0, f(x0), "ro")
plt.grid(True)
plt.xlabel("x", fontsize=14)
plt.ylabel("f(x)", fontsize=14, rotation=0)
plt.axis([-2, 2, -5, 15]);





learning_rate = 0.01
x = tf.Variable(0.)

for i, epoch in enumerate(range(150)):
    with tf.GradientTape() as tape:
        z = f(x)
    dz_dx = tape.gradient(z, x)
    print('Epoch:', i, 'Grad:', dz_dx.numpy())
    x.assign_sub(learning_rate * dz_dx)





x.numpy()





f(x.numpy())





x = tf.Variable(0.)
optimizer = tf.keras.optimizers.SGD(lr=0.01)

for iteration in range(150):
    with tf.GradientTape() as tape:
        z = f(x)
    dz_dx = tape.gradient(z, x)
    print('Epoch:', i, 'Grad:', dz_dx.numpy())
    optimizer.apply_gradients([(dz_dx, x)])





x.numpy()


# ### Building a basic Image Classifier using DNN
# We will keep things simple here with regard to the key objective. We will build a simple apparel classifier by training models on the very famous [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) based on Zalando’s article images consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. The idea is to classify these images into an apparel category amongst 10 categories on which we will be training our models on.
# 
# Here's an example how the data looks (each class takes three-rows):
# 
# <img src="https://s3-eu-central-1.amazonaws.com/zalando-wp-zalando-research-production/2017/08/fashion-mnist-sprite.png">
# <i>Source:https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/</i>
# 
# Fashion MNIST is intended as a drop-in replacement for the classic [MNIST dataset](http://yann.lecun.com/exdb/mnist/) often used as the "Hello, World" of machine learning programs for computer vision. You can access the Fashion MNIST directly from TensorFlow.
# 
# __Note__: Although these are really images, they are loaded as NumPy arrays and not binary image objects.




# get the dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train.shape, X_test.shape





# scale pixel densities
X_train = X_train / 255.
X_test = X_test / 255.





# visualize a sample datapoint
plt.imshow(X_train[0], cmap="binary");





# humanise class labels
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]





# plot a few examples
n_rows = 4
n_cols = 10
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
        plt.axis('off')
        plt.title(class_names[y_train[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5);





# set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# ## Build a 2 Layer DNN
# 
# Build a Sequential model (``tf.keras.models.Sequential``) and add four layers to it:
# 
# + a Flatten layer (``tf.keras.layers.Flatten``) to convert each $28x28$ image to a single row of $784$ pixel values. Since it is the first layer in your model, you should specify the input_shape argument, leaving out the batch size: $[28, 28]$.
# + a Dense layer (``tf.keras.layers.Dense``) with $300$ neurons (aka units), and the __"relu"__ activation function.
# + Another Dense layer with $100$ neurons, also with the __"relu"__ activation function.
# + A final Dense layer with $10$ neurons (one per class), and with the __"softmax"__ activation function to ensure that the sum of all the estimated class probabilities for each image is equal to $1$.




model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10, activation="softmax")
                                    
])





# compile the model
model.compile(loss="sparse_categorical_crossentropy", 
              optimizer=tf.keras.optimizers.SGD(1e-3),
              metrics=["accuracy"])
model.summary()





# train/fit the model
history = model.fit(X_train, y_train, 
                    epochs=20,
                    batch_size=32,
                    validation_split=0.1)





# plot training progress
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0.3, 1);





# get predictions
y_pred = model.predict(X_test)


# Often, you may only be interested in the most likely class.
# 
# Use ``np.argmax()`` to get the class ID of the most likely class for each instance. Tip: you want to set ``axis=1``.




y_pred = y_pred.argmax(axis=1)





# evaluate model
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred,
      target_names=class_names))


# ## Batch Normalization
# 
# Batch normalization is a technique for training deep neural networks that standardizes the inputs for each mini-batch before going into a layer. This has the effect of stabilizing the learning process and preventing overfitting.
# 
# + Fixed distributions of inputs would remove the ill effects of the internal covariate shift (change in the distributions of neurons when training) [https://arxiv.org/abs/1502.03167]
# 
# + Batch normalization acts to standardize only the mean and variance of each unit in order to stabilize learning [https://amzn.to/2NJW3gE]

# ## Build a model with Batch-Normalization




model = tf.keras.models.Sequential([            
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    # batch-norm layer
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(300, 
                          activation='relu'),
    # batch-norm layer
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(100, 
                          activation='relu'),
    # batch-norm layer
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation="softmax")
])





# compile model
model.compile(loss="sparse_categorical_crossentropy", 
              optimizer=tf.keras.optimizers.SGD(1e-3),
              metrics=["accuracy"])
model.summary()





# fit/train the model
history = model.fit(X_train, y_train, epochs=20,
                    validation_split=0.1)





# visualize training progress
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0.3, 1);





# evaluate model performance
y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=1)
print(classification_report(y_test, y_pred,
      target_names=class_names))


# ### Changing the position of Batchnorm
# Sometimes applying BN before the activation function works better (there's a debate on this topic). Moreover, the layer before a BatchNormalization layer does not need to have bias terms, since the BatchNormalization layer already includes it, hence it would be a waste of parameters, so you can set ``use_bias=False`` when creating those layers
# 
# Source: https://www.google.co.in/books/edition/Hands_On_Machine_Learning_with_Scikit_Le/bRpYDgAAQBAJ

# ## Dropout
# 
# DNNs have a large number of parameters. However, overfitting is a serious problem in such networks. Large networks are also slow to train and use, making it difficult to deal with overfitting by combining the predictions of many different large neural nets at test time. Dropout is a technique to randomly drop units (along with their connections) from the neural network during training. This prevents units from co-adapting too much. 
# 
# Refer to the paper from Hinton et. al. discussing this in detail: [paper](https://jmlr.org/papers/v15/srivastava14a.html)




# define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(300, activation="elu", kernel_initializer="glorot_uniform"),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(100, activation="elu", kernel_initializer="glorot_uniform"),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(10, activation="softmax")
])

# compile model
model.compile(loss="sparse_categorical_crossentropy", 
              optimizer="adam", metrics=["accuracy"])


# train the model
history = model.fit(X_train, y_train, epochs=20,
                    validation_split=0.1)





# evaluate model
y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=1)
print(classification_report(y_test, y_pred,
      target_names=class_names))

