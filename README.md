# NeuroNER

[![Build Status](https://travis-ci.org/Franck-Dernoncourt/NeuroNER.svg?branch=master)](https://travis-ci.org/Franck-Dernoncourt/NeuroNER)

NeuroNER is a program that performs named-entity recognition (NER). Website: [neuroner.com](http://neuroner.com).

This page gives step-by-step instructions to install and use NeuroNER. If you already have Python 3.5 and TensorFlow 1.0, you can directly jump to the [Downloading NeuroNER](#downloading-neuroner).


## Table of Contents

<!-- toc -->

- [Installing NeuroNER](#installing-neuroner)
  * [Requirements](#requirements)
  * [Downloading NeuroNER](#downloading-neuroner)
- [Using NeuroNER](#using-neuroner)
  * [Adding a new dataset](#adding-a-new-dataset)
  * [Using a pretrained model](#using-a-pretrained-model)
  * [Sharing a pretrained model](#sharing-a-pretrained-model)
  * [Using TensorBoard](#using-tensorboard)
- [Citation](#citation)

<!-- tocstop -->

## Installing NeuroNER

### Requirements

NeuroNER relies on Python 3.5, TensorFlow 1.0+, and optionally on BRAT:

- Python 3.5: NeuroNER does not work with Python 2.x. On Windows, it has to be Python 3.5 64-bit.
- TensorFlow is a library for machine learning. NeuroNER uses it for its NER engine, which is based on neural networks. Official website: [https://www.tensorflow.org](https://www.tensorflow.org)
- BRAT (optional) is a web-based annotation tool. It only needs to be installed if you wish to conveniently create annotations or view the predictions made by NeuroNER. Official website: [http://brat.nlplab.org](http://brat.nlplab.org)

Installation instructions for TensorFlow, Python 3.5, and (optional) BRAT are given below for different types of operating systems:

- [Mac](install_mac.md)
- [Ubuntu](install_ubuntu.md)
- [Windows](install_windows.md)


Alternatively, you can use this [installation script](install_ubuntu.sh) for Ubuntu, which:

1. Installs TensorFlow (CPU only) and Python 3.5.
2. Downloads the NeuroNER code as well as the word embeddings.
3. Starts training on the CoNLL-2003 dataset (the F1-score on the test set should be around 0.90, i.e. on par with state-of-the-art systems).

To use this script, run the following command from the terminal:

```
wget https://raw.githubusercontent.com/Franck-Dernoncourt/NeuroNER/master/install_ubuntu.sh; bash install_ubuntu.sh
```


## Downloading NeuroNER

To download NeuroNER code, download and unzip https://github.com/Franck-Dernoncourt/NeuroNER/archive/master.zip, which can be done on Ubuntu and Mac OS X with:

```
wget https://github.com/Franck-Dernoncourt/NeuroNER/archive/master.zip
sudo apt-get install -y unzip # This line is for Ubuntu users only
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

NeuroNER is now ready to run.



## Using NeuroNER

By default NeuroNER is configured to train and test on the CoNLL-2003 dataset. To start the training:

```
# To use the CPU if you have installed tensorflow, or use the GPU if you have installed tensorflow-gpu:
python3.5 main.py

# To use the CPU only if you have installed tensorflow-gpu:
CUDA_VISIBLE_DEVICES="" python3.5 main.py

# To use the GPU 1 only if you have installed tensorflow-gpu:
CUDA_VISIBLE_DEVICES=1 python3.5 main.py
```

If you wish to change any of NeuroNER parameters, you should modify the [`src/parameters.ini`](src/parameters.ini) configuration file. Alternatively, any parameter may be specified in the command line.

For example, to reduce the number of training epochs and not use any pre-trained token embeddings:
```
python3.5 main.py --maximum_number_of_epochs=2 --token_pretrained_embedding_filepath=""
```


To perform NER on some plain texts using a pre-trained model:

```
python3.5 main.py --train_model=False --use_pretrained_model=True --dataset_text_folder=../data/example_unannotated_texts --pretrained_model_folder=../trained_models/conll_2003_en
```

If a parameter is specified in both the [`src/parameters.ini`](src/parameters.ini) configuration file and as a command line argument, then the command line argument takes precedence (i.e., the parameter in [`src/parameters.ini`](src/parameters.ini) is ignored). You may specify a different configuration file with the `--parameters_filepath` command line argument. The command line arguments have no default value except for `--parameters_filepath`, which points to [`src/parameters.ini`](src/parameters.ini).

NeuroNER has 3 modes of operation:

- training mode (from scratch): the dataset folder must have train and valid sets. Test and deployment sets are optional.
- training mode (from pretrained model): the dataset folder must have train and valid sets. Test and deployment sets are optional.
- prediction mode (using pretrained model): the dataset folder must have either a test set or a deployment set.

### Adding a new dataset

A dataset may be provided in either CoNLL-2003 or BRAT format. The dataset files and folders should be organized and named as follows:

- Training set: `train.txt` file (CoNLL-2003 format) or `train` folder (BRAT format). It must contain labels.
- Validation set: `valid.txt` file (CoNLL-2003 format) or `valid` folder (BRAT format). It must contain labels.
- Test set: `test.txt` file (CoNLL-2003 format) or `test` folder (BRAT format). It must contain labels.
- Deployment set: `deploy.txt` file (CoNLL-2003 format) or `deploy` folder (BRAT format). It shouldn't contain any label (if it does, labels are ignored).

We provide several examples of datasets:

- [`data/conll2003/en`](data/conll2003/en): annotated dataset with the CoNLL-2003 format, containing 3 files (`train.txt`, `valid.txt` and  `test.txt`).
- [`data/example_unannotated_texts`](data/example_unannotated_texts): unannotated dataset with the BRAT format, containing 1 folder (`deploy/`). Note that the BRAT format with no annotation is the same as plain texts.



### Using a pretrained model

In order to use a pretrained model, the `pretrained_model_folder` parameter in the [`src/parameters.ini`](src/parameters.ini) configuration file must be set to the folder containing the pretrained model. The following parameters in the [`src/parameters.ini`](src/parameters.ini) configuration file must also be set to the same values as in the configuration file located in the specified `pretrained_model_folder`:

```
use_character_lstm
character_embedding_dimension
character_lstm_hidden_state_dimension
token_pretrained_embedding_filepath
token_embedding_dimension
token_lstm_hidden_state_dimension
use_crf
tagging_format
tokenizer
```


### Sharing a pretrained model

You are highly encouraged to share a model trained on their own datasets, so that other users can use the pretrained model on other datasets. We provide the [`src/prepare_pretrained_model.py`](src/prepare_pretrained_model.py) script to make it easy to prepare a pretrained model for sharing. In order to use the script, one only needs to specify the `output_folder_name`, `epoch_number`, and `model_name` parameters in the script.

By default, the only information about the dataset contained in the pretrained model is the list of tokens that appears in the dataset used for training and the corresponding embeddings learned from the dataset.

If you wish to share a pretrained model without providing any information about the dataset (including the list of tokens appearing in the dataset), you can do so by setting

```delete_token_mappings = True```

when running the script. In this case, it is highly recommended to use some external pre-trained token embeddings and freeze them while training the model to obtain high performance. This can be done by specifying the `token_pretrained_embedding_filepath` and setting

```freeze_token_embeddings = True```

in the [`src/parameters.ini`](src/parameters.ini) configuration file during training.

In order to share a pretrained model, please [submit a new issue](https://github.com/Franck-Dernoncourt/NeuroNER/issues/new) on the GitHub repository.

### Using TensorBoard

You may launch TensorBoard during or after the training phase. To do so, run in the terminal from the NeuroNER folder:
```
tensorboard --logdir=output
```

This starts a web server that is accessible at http://127.0.0.1:6006 from your web browser.

## Citation

If you use NeuroNER in your publications, please cite this [paper](https://arxiv.org/abs/1705.05487):

```
@article{2017neuroner,
  title={{NeuroNER}: an easy-to-use program for named-entity recognition based on neural networks},
  author={Dernoncourt, Franck and Lee, Ji Young and Szolovits, Peter},
  journal={Conference on Empirical Methods on Natural Language Processing (EMNLP)},
  year={2017}
}
```

The neural network architecture used in NeuroNER is described in this [article](https://arxiv.org/abs/1606.03475):

```
@article{2016deidentification,
  title={De-identification of Patient Notes with Recurrent Neural Networks},
  author={Dernoncourt, Franck and Lee, Ji Young and Uzuner, Ozlem and Szolovits, Peter},
  journal={Journal of the American Medical Informatics Association (JAMIA)},
  year={2016}
}
```
