rsm
===

A Python implementation of the Replicated Softmax model.

This is an implementation of R. Salakhutdinov and G.E. Hinton's Replicated Softmax model,
see [http://www.mit.edu/~rsalakhu/papers/repsoft.pdf](http://www.mit.edu/~rsalakhu/papers/repsoft.pdf)

You will need Numpy and SciPy installed. I don't provide the dataset.
Also feel free to change the initialization of the biases, it just worked for me.

It is part of my [M.Sc. Thesis "Image Object and Document Classification using Neural Networks with Replicated Softmax Input Layers"](http://www.fylance.de/msc/), advised by Christian Osendorfer at [TUM](http://www.tum.de), [I6](http://www6.in.tum.de/).

- rsm.py: contains the RSM implementation encapsulated in a class.
- rsm_driver_20newsgroups.py: shows an exemplary usage of the class.
