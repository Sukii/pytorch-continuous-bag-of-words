# continuous-bag-of-words
The Continuous Bag-of-Words model (CBOW) is frequently used in NLP deep learning. It is a model that tries to predict words given the context of a few words before and a few words after the target word.
This is distinct from language modeling, since CBOW is not sequential and does not have to be probabilistic. Typically, CBOW is used to quickly train word embeddings, and these embeddings are used to initialize the embeddings of some more complicated model. Usually, this is referred to as pretraining embeddings. It almost always helps performance a couple of percent.

This is the solution of the final exercise of [this](http://pytorch.org/tutorials/beginner/deep_learning_nlp_tutorial.html) great tutorial on NLP in PyTorch.

     .../anaconda3/bin/python3 cbow.py sample.txt 