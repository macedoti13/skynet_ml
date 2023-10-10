# SkyNet: A Numpy-powered, 100% Hand Made, Machine Learning Library 🚀

Welcome to SkyNet! This is my personal machine learning library. It is entirely made from scratch, using only NumPy. No professional machine learning libraries like Scikit-Learn, TensorFlow, or PyTorch are allowed. Everything is made by applying the fundamental concepts of machine learning. It supports both classical algorithms, supervised and unsupervised, as well as deep learning. This project is in its first stages, so there is a lot more to come. This serves as a showcase of my skills and conceptual knowledge in machine learning, calculus, linear algebra, and statistics. Stay tuned, because there is a lot more to be implemented here!

![Banner/Image](images/skynet.png)

> **Disclaimer**: SkyNet is, at its heart, a project of passion and learning. Please refrain from deploying it in a professional setting. While meticulously crafted, it doesn't leverage state-of-the-art optimization and lacks GPU support.

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Acknowledgements](#acknowledgements)

## Features
- **NumPy Powered:** SkyNet is built entirely with NumPy, showcasing a true understanding of the algorithms.
- **Classical Machine Learning:** SkyNet supports both supervised and unsupervised machine learning models.
- **Custom Neural Nets:** SkyNet allows you to create custom neural networks with customizable layers, multiple activation functions, and optimization methods.
- **Transparency:** SkyNet's code is well-documented with detailed comments and docstrings to guide you through each part of the code.

## Installation

Install `skynet_ml` directly from PyPI:

```bash
pip install skynet-ml
```

## Usage
Getting started with SkyNet is pretty straightforward. Here's a step-by-step guide:

1. **Setup Your Environment:**

Before diving in, ensure you have numpy:
```bash
pip install numpy
```

2. **Enter skynet:** 

Navigate to the SkyNet directory:
```bash
cd skynet
```

3. **Make Magic!:** 
```bash
# import all the stuff you need 
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from skynet_ml.nn.models import Sequential
from skynet_ml.nn.layers import Dense
from skynet_ml.nn.optimizers import Adam
from skynet_ml.metrics import ConfusionMatrix
from skynet_ml.utils.nn.early_stopping import EarlyStopping

# read the mnist dataset
(x_train, y_train),(x_test, y_test) = mnist.load_data()

num_labels = len(np.unique(y_train))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
image_size = x_train.shape[1]
input_size = image_size**2

x_train = np.reshape(x_train, [-1, input_size])
x_train = x_train.astype('float32') / 255
x_test = np.reshape(x_test, [-1, input_size])
x_test = x_test.astype('float32') / 255

# create the model
model = Sequential()

# add layers to the mdoel
model.add(Dense(n_units=150, activation="leaky_relu", input_dim=input_size))
model.add(Dense(n_units=150, activation="leaky_relu"))
model.add(Dense(n_units=num_labels, activation="softmax"))

# compile the model
opt = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
model.compile(loss="categorical_crossentropy", optimizer=opt, regularizer="l2")

# fit your model
model.fit(x_train=x_train, 
          y_train=y_train, 
          x_val=x_test, 
          y_val=y_test, 
          metrics=["accuracy", "precision", "recall", "fscore"], 
          early_stopping=EarlyStopping(patience=10, min_delta=0.0001),
          epochs=5, 
          batch_size=32,
          save_training_history_in="testing/logs/mnist_sequential.csv")

# predict with the model
y_pred = model.predict(x_test)

# compute the confusion matrix
cf = ConfusionMatrix(task_type="multiclass").compute(y_test, y_pred)
ConfusionMatrix(task_type="multiclass").plot(cf, save_in="testing/logs/confusion_matrix.png")

# save the model 
save_model(model, "testing/logs/mnist_sequential.pkl")
plot_model(model, save_in="testing/logs/mnist_sequential_model.txt")

# plot the training history
plot_training_history("testing/logs/mnist_sequential.csv", save_in="testing/logs/mnist_sequential.png")
```

## Acknowledgements

- Big thanks to Numpy for being the foundation of this project.
- Grateful for my professor Lucas Kupssinsku who is teaching me all this stuff. 
- Big thanks to Ian Goodfellow for writing a bible about deep learning. 
- Big thanks to my boy chatGPT, who wrote 90% of the docstrings because I'm way to lazzy (including this docstring hehe).

![Banner/Image](images/terminator.png)

> **Disclaimer**: SkyNet is a personal project and it's not designed for professional use. Therefore, do not sue me for using the same name as the evil A.I in The Terminator, the name is a joke. 