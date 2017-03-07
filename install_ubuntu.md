# Installing NeuroNER on Ubuntu

## Ubuntu 14.04 and 16.04

If you use Ubuntu 14.04, you need to install Python 3.5 (by default Ubuntu 14.04 doesn't have Python 3.5, unlike Ubuntu 16.04), which you can do as follows:

```
sudo add-apt-repository ppa:fkrull/deadsnakes
sudo apt-get update
sudo apt-get install -y python3.5
wget https://bootstrap.pypa.io/get-pip.py
sudo python3.5 get-pip.py
pip3 install --upgrade pip
pip -V
sudo mv  /usr/local/bin/pip /usr/local/bin/pip3
sudo ln -s /usr/local/bin/pip2.7 /usr/local/bin/pip
```

To install TensorFlow:
```
# If you want NeuroNER to be able to use the GPU (in which case CUDA >= 8.0 is required):
sudo pip3 install tensorflow-gpu

# If you do not plan to use the GPU:
sudo pip3 install tensorflow
```

To install CUDA 8.0 (only required if you have installed `tensorflow-gpu`):

```
wget  https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1404-8-0-local-ga2_8.0.61-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1404-8-0-local-ga2_8.0.61-1_amd64-deb
sudo apt-get update
sudo apt-get install cuda
```

To install a few more packages which NeuroNER depends on:

```
sudo pip3 install -U networkx matplotlib scikit-learn scipy
```

To install BRAT (optional, you just need it if you want to easily change or view the annotations):

```
Installing brat:
sudo apt install -y apache2
mkdir brat
cd brat
mkdir -p data work
wget http://weaver.nlplab.org/~brat/releases/brat-v1.3_Crunchy_Frog.tar.gz
tar xzf brat-v1.3_Crunchy_Frog.tar.gz
cd brat-v1.3_Crunchy_Frog
./install.sh -u
python standalone.py
```

BRAT should now be accessible through the browser at [127.0.0.1:8001](127.0.0.1:8001).
