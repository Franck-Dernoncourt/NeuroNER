# Installing NeuroNER on Mac OS X


You need to install Python 3.5:

download and install from https://www.python.org/ftp/python/3.5.3/python-3.5.3-macosx10.6.pkg.

To install TensorFlow:
```
# For CPU support (no GPU support):
sudo pip3 install tensorflow
# For GPU support:
sudo pip3 install tensorflow-gpu
```

Note that for GPU support, [GPU requirements for Tensorflow](https://www.tensorflow.org/install/install_mac) must be satisfied.

To install a few more packages which NeuroNER depends on:

```
sudo pip3 install -U networkx matplotlib scikit-learn scipy spacy pycorenlp
python3.5 -m spacy download en
```

To install BRAT (optional, you just need it if you want to easily change or view the annotations):

```
Installing brat:
mkdir brat
cd brat
wget http://weaver.nlplab.org/~brat/releases/brat-v1.3_Crunchy_Frog.tar.gz
tar xzf brat-v1.3_Crunchy_Frog.tar.gz
cd brat-v1.3_Crunchy_Frog
./install.sh -u
python standalone.py
```

BRAT should now be accessible through the web browser at [http://127.0.0.1:8001](http://127.0.0.1:8001).
