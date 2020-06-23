# GENERAL BASIC CONVOLUTIONAL NEURAL NETWORK

This is the source code of basic image classifier using CNN. This is part of the Bangkit [Bangkit](https://events.withgoogle.com/bangkit/) program final project. The framework is designed to ease the process of construction of model architecture and training from image dataset.

## Overview

The framework of the model is using Keras module of Tensorflow v2.2.0. A general class of CNN equipped with some specific methods is created to simplify the process of machine learning model development. The framework is divided to two classes, the one that accomodate transfer learning and the one to build from scratch. All class frameworks are contained in `model` module.

Some Jupyter notebooks with specific case (plant disease dataset) are also created to allow anyone to try the framework without cloning the repository. 

## Getting started

### To use the framework directly using notebooks

1. Open the [notebook](https://github.com/phoenixfin/general-cnn/tree/master/notebook) directory in this repository

2. Open one of the file.

3. Click `open in colab` icon in the top of the code.

4. Follow all instruction in the notebook.

5. Enjoy!

### To run the frameworks locally

1. Clone this repo

   ```
   git clone https://github.com/phoenixfin/general-cnn
   ```

2. Create virtual environment and activate

   OSX / Ubuntu :

   ```
   python3 -m venv venv
   source ./venv/bin/activate
   ```

   Windows :

   ```
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install Tensorflow 2.2 (all other package requirements are default, e.g. matplotlib, os, numpy) 

   ```
   pip3 install tensorflow
   ```

4. Write some codes in the `main.py` as needed. Some examples are provided. 

5. Run it!

   ```
   python main.py
   ```
