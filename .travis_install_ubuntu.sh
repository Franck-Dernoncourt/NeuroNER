#!/bin/bash
ls -la
pip -V
pip3 -V
free -m
vmstat -s
dmidecode -t 17

sudo apt-get install -y unzip
mkdir data/word_vectors
cd data/word_vectors

# Download some word embeddings
#wget http://nlp.stanford.edu/data/glove.6B.zip
wget http://neuroner.com/data/word_vectors/glove.6B.100d.zip
unzip glove.6B.100d.zip

# Going back to NeuroNER src folder
ls -la
cd ../../src
ls -la