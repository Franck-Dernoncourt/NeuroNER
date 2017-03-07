# NeuroNER

NeuroNER is a program that performs named-entity recognition (NER). Website: [neuroner.com](http://neuroner.com).

This page gives step-by-step instructions to install and use NeuroNER. If you already have Python 3.5 and TensorFlow 1.0, you can directly jump to the [Running NeuroNER section](#running-neuroner).

Alternatively, you can use this [installation script](install_ubuntu.sh) for Ubuntu, which:

1. Installs TensorFlow (CPU only), Python 3.5, and BRAT.
2. Downloads the NeuroNER code as well as word embeddings.
3. Starts training on the CoNLL-2003 dataset (the F1-score on the test set should be around 0.90, i.e. on par with state-of-the-art systems).

To use this script, run the following command from the terminal:

```
wget https://raw.githubusercontent.com/Franck-Dernoncourt/NeuroNER/master/install_ubuntu.sh; bash install_ubuntu.sh
```

## Installation

### Requirements

NeuroNER relies on BRAT, Python 3.5, and TensorFlow 1.0:

- BRAT is a web-based annotation tool. It only needs to be installed if you wish to conveniently create annotations or view the predictions made by NeuroNER. Official website: [http://brat.nlplab.org](http://brat.nlplab.org)
- Python 3.5: NeuroNER does not work with Python 2.x.
- TensorFlow is a library for machine learning. NeuroNER uses it for its NER engine, which is based on neural networks. Official website: [https://www.tensorflow.org/](https://www.tensorflow.org/)

Installation instructions for TensorFlow, Python 3.5, and (optional) BRAT are given below for different types of operating systems:

- [Mac](install_mac.md)
- [Ubuntu](install_ubuntu.md)
- [Windows](install_windows.md)


###Running NeuroNER

To download NeuroNER code, download and unzip `http://neuroner.com/NeuroNER-master.zip`, which can be done on Ubuntu and Mac OS X with:

```
wget https://github.com/Franck-Dernoncourt/NeuroNER/archive/master.zip
sudo apt-get install -y unzip
unzip master.zip
```

It also needs some word embeddings, which should be downloaded from http://neuroner.com/data/word_vectors/glove.6B.100d.zip, unzipped and placed in `/data/word_vectors`. This can be done on Ubuntu and Mac OS X with:

```
# Download some word embeddings
mkdir NeuroNER-master/data/word_vectors
cd NeuroNER-master/data/word_vectors
wget http://neuroner.com/data/word_vectors/glove.6B.100d.zip
unzip glove.6B.100d.zip
```

NeuroNER is now ready to run! By default NeuroNER is configured to train and test on CoNLL-2003. To start the training:

```
# To use the CPU if you have installed tensorflow, or use the GPU if you have installed tensorflow-gpu:
python3.5 main.py

# To use the CPU only if you have installed tensorflow-gpu:
CUDA_VISIBLE_DEVICES="" python3.5 main.py

# To use the GPU 1 only if you have installed tensorflow-gpu:
CUDA_VISIBLE_DEVICES=1 python3.5 main.py
```


## Using NeuroNER

If you wish to change the dataset or NeuroNER parameters, you should change the [`src/parameters.ini`](src/parameters.ini) configuration file.
