# Installing NeuroNER requirements on Mac OS X


You need to install Python 3.5 (e.g., from https://www.python.org/ftp/python/3.5.3/python-3.5.3-macosx10.6.pkg)

To install TensorFlow:
```
# For CPU support (no GPU support):
sudo pip3 install tensorflow
# For GPU support:
sudo pip3 install tensorflow-gpu
```

Note that for GPU support, [GPU requirements for Tensorflow](https://www.tensorflow.org/install/install_mac) must be satisfied.

To install a few more Python packages which NeuroNER depends on:

```
sudo pip3 install -U networkx matplotlib scikit-learn scipy spacy pycorenlp
python3.5 -m spacy download en
```

To install BRAT (optional, you just need it if you want to easily create, change or view the annotations):

```
mkdir brat
cd brat
wget http://weaver.nlplab.org/~brat/releases/brat-v1.3_Crunchy_Frog.tar.gz
tar xzf brat-v1.3_Crunchy_Frog.tar.gz
cd brat-v1.3_Crunchy_Frog
./install.sh -u

# To run BRAT (requires Python 2.5, 2.6 or 2.7):
python standalone.py
```

BRAT should now be accessible through the web browser at [http://127.0.0.1:8001](http://127.0.0.1:8001).

You can now [download and run NeuroNER](README.md#downloading-neuroner).
