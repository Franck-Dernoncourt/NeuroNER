# Installing NeuroNER on Microsoft Windows

(tested on Windows 7 SP1 64-bit)

## Table of Contents

<!-- toc -->

- [Python 3.5, TensorFlow, and several other Python packages](#python-35-and-tensorflow)
- [SpaCy](#spacy)
- [Perl](#perl)
- [TensorBoard](#tensorboard)

<!-- tocstop -->

## Python 3.5, TensorFlow, and several other Python packages
First, install Python 3.5 and TensorFlow following these instructions: [How to install TensorFlow on Windows?](http://stackoverflow.com/a/39902815/395857)

Then, from the command prompt (make sure `pip` is connected to Python 3.5. You can verify it by running `pip -V`):

```
pip install -U networkx matplotlib scikit-learn scipy pycorenlp
```

## SpaCy
You need to install SpaCy. To do so, two solutions.

Solution 1: Installing Visual Studio Express 2015 (https://www.visualstudio.com/vs/visual-studio-express, free but takes 12 GB of space on the hard drive), then run:
```
pip install -U spacy
python -m spacy.en.download
```

Solution 2: Use Anaconda, in which case there is no need to install Visual Studio Express 2015:

```
conda config --add channels conda-forge
conda install spacy
python -m spacy.en.download
python -m spacy download en

```

## Perl
You also need to install Perl (because the official CoNLL-2003 evaluation script is written in Perl): http://strawberryperl.com

Make sure the `perl.exe` binary is in your `Path` system environment variable:

![](http://neuroner.com/perl2.png "")

<!---

To add perl in your `Path` system environment variable:

![](http://neuroner.com/perl.png "")

!-->

## BRAT (optional)

Installing BRAT is optional: it is only needed to easily create, change or view the annotations. BRAT does not run natively on Microsoft Windows, however it runs smoothly on [Cygwin](https://www.cygwin.com). After installing Cygwin, run `Cygwin.bat` (by default located on `C:\cygwin64\` if you used the 64-bit installation for Cygwin). Then, in the terminal that just opened, run:

```
mkdir brat
cd brat
wget http://weaver.nlplab.org/~brat/releases/brat-v1.3_Crunchy_Frog.tar.gz
tar xzf brat-v1.3_Crunchy_Frog.tar.gz
cd brat-v1.3_Crunchy_Frog
./install.sh -u
python standalone.py
```

BRAT should now be accessible through the web browser at [http://127.0.0.1:8001](http://127.0.0.1:8001).



## TensorBoard (optional)
 The `tensorboard.exe` binary should also be in your `Path` system environment variable, if you plan to use TensorBoard (optional).
