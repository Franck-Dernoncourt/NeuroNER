#!/bin/bash

#  If you use Ubuntu 14.04, you need to install Python 3.5 (by default Ubuntu 14.04 doesn't have Python 3.5, unlike Ubuntu 16.04):
sudo add-apt-repository -y ppa:fkrull/deadsnakes
sudo apt-get update
sudo apt-get install -y python3.5
wget https://bootstrap.pypa.io/get-pip.py
sudo python3.5 get-pip.py
pip3 install --upgrade pip
pip -V
sudo mv  /usr/local/bin/pip /usr/local/bin/pip3
sudo ln -s /usr/local/bin/pip2.7 /usr/local/bin/pip

# To install TensorFlow:
sudo pip3 install tensorflow

# To install a few more packages which NeuroNER depends on:
sudo pip3 install -U networkx matplotlib scikit-learn scipy pycorenlp

# Installing spaCy
sudo apt-get install -y build-essential python3.5-dev
sudo pip3 install -U spacy
sudo python3.5 -m spacy download en

# To install NeuroNER:
wget https://github.com/Franck-Dernoncourt/NeuroNER/archive/master.zip
sudo apt-get install -y unzip
unzip master.zip
mkdir NeuroNER-master/data/word_vectors
cd NeuroNER-master/data/word_vectors

# Download some word embeddings
#wget http://nlp.stanford.edu/data/glove.6B.zip
wget http://neuroner.com/data/word_vectors/glove.6B.100d.zip
unzip glove.6B.100d.zip

# NeuroNER is now ready to run! By default it is configured to train and test on CoNLL-2003. To start the training:
# To use the CPU if you have installed tensorflow, or use the GPU if you have installed tensorflow-gpu:
cd ../../src
python3.5 main.py
