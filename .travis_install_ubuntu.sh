#!/bin/bash
ls -la
pip -V
pip3 -V
free -m
vmstat -s
dmidecode -t 17

sudo apt-get install -y unzip

# download some word embeddings
mkdir ./data/word_vectors
wget -P data/word_vectors http://neuroner.com/data/word_vectors/glove.6B.100d.zip
unzip data/word_vectors/glove.6B.100d.zip -d data/word_vectors/