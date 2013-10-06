"""
@author:  Joerg Landthaler
@credits: Christian Osendorfer
@date:    Nov, 2011
@organization: TUM, I6, Machine Learning Group
@summary: Implementation of the Replicated Softmax model,
          as presented by R. Salakhutdinov & G.E. Hinton
          in http://www.mit.edu/~rsalakhu/papers/repsoft.pdf
@version: 1.0
@
"""
import scipy as sp
import numpy as np

class RSM(object):
    def train(self, data, units, epochs=1000, lr=0.0001, weightinit=0.001, 
            momentum=0.9, btsz=100):
        """
        Standard CD-1 training.
        @param data: a (rowwise) sample matrix. Number of samples should be divisible by btsz.
        @param units: #latent topics
        @param epochs: #training epochs
        @param lr: learning rate
        @param weightinit: scaling of random weight initialization
        @param momentum: momentum rate
        @param btsz: batchsize   
        """
        print
        print "[RSM_NUMPY] Training with CD-1, hidden:", units
        dictsize = data.shape[1]
        # initilize weights
        w_vh = weightinit * np.random.randn(dictsize, units)
        w_v = weightinit * np.random.randn(dictsize)
        w_h = weightinit * np.random.randn(units)
        # weight updates
        wu_vh = np.zeros((dictsize, units))
        wu_v = np.zeros((dictsize))
        wu_h = np.zeros((units))
        delta = lr/btsz
        batches = data.shape[0]/btsz
        print "learning_rate: %f"%delta
        print "updates per epoch: %s | total updates: %s"%(batches, batches*epochs)
        err_list = []
        for epoch in xrange(epochs):
            print "[RSM_NUMPY] Epoch", epoch
            err = []
            for b in xrange(batches):
                start = b * btsz 
                v1 = data[start : start+btsz]
                # hidden biases scaling factor (D in paper)
                D = v1.sum(axis=1)
                # calculate hidden activations
                h1 = sigmoid((np.dot(v1, w_vh) + np.outer(D, w_h)))
                # sample hiddens
                h_rand = np.random.rand(btsz, units)
                h_sampled = np.array(h_rand < h1, dtype=int)
                # calculate visible activations
                v2 = np.dot(h_sampled, w_vh.T) + w_v
                tmp = np.exp(v2)
                sum = tmp.sum(axis=1)
                sum = sum.reshape((btsz,1))
                v2_pdf = tmp/sum
                # sample D times from multinomial
                v2 *= 0
                for i in xrange(btsz):
                    v2[i] = np.random.multinomial(D[i],v2_pdf[i],size=1)              
                # use activations, not sampling here
                h2 = sigmoid(np.dot(v2, w_vh) + np.outer(D, w_h))
                # compute updates
                wu_vh = wu_vh * momentum + np.dot(v1.T, h1) - np.dot(v2.T, h2)
                wu_v = wu_v * momentum + v1.sum(axis=0) - v2.sum(axis=0)
                wu_h = wu_h * momentum + h1.sum(axis=0) - h2.sum(axis=0)
                # update 
                w_vh += wu_vh * delta 
                w_v += wu_v * delta
                w_h += wu_h * delta
                # calculate reconstruction error
                err.append(np.linalg.norm(v2-v1)**2/(dictsize*btsz))
            mean = np.mean(err)
            print "Mean squared error: " + str(mean)
            err_list.append(float(mean))
        return {"w_vh": w_vh, 
                "w_v": w_v, 
                "w_h": w_h,
                "err": err_list}
def sigmoid(X):
    """
    sigmoid of X
    """
    return (1 + sp.tanh(X/2))/2