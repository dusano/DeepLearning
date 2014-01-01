DeepLearning
============

Python implementation of UFLDL tutorial code (http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial)

Installation
------------

1. Clone repository
2. Set up virtualenv
3. pip install -r requirements.txt

To speed things up, install Intel Math Kernel Library and fill-in ~/.numpy-site.cfg before installing numpy (see http://stackoverflow.com/questions/13769936/supplying-numpy-site-cfg-arguments-to-pip for more information).

Test
----
Running

> python stacked_autoencoder_test.py

should produce

> Before Fine-tuning Test Accuracy: 92.180%
>
> After Fine-tuning Test Accuracy: 97.830%

on MNIST data set (http://yann.lecun.com/exdb/mnist/)

Enjoy!
