# Installing NeuroNER on Microsoft Windows

(tested on Windows 7 SP1 64-bit)

First, install Python 3.5 and TensorFlow following these instructions: [How to install TensorFlow on Windows?](http://stackoverflow.com/a/39902815/395857)

Then, from the command prompt (make sure `pip` is connected to Python 3.5. You can do so by running `pip -V`):

```
pip install -U networkx matplotlib scikit-learn scipy
pip install --upgrade tensorflow-gpu
```

You also need to install Perl (because the official CoNLL-2003 evaluation script is written in Perl): http://strawberryperl.com

Make sure the `perl.exe` binary is in your `Path` system environment variable:

![](http://neuroner.com/perl2.png "")

To add perl in your `Path` system environment variable:

![](http://neuroner.com/perl.png "")

