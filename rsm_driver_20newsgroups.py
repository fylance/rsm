"""
Train RSM on 20 Newsgroups dataset (wordcounts)
"""
import rsm_numpy, time, cPickle

# hyperparameters
hiddens = 50
batchsize = 100
epochs = 1000
learning_rate = 0.0001

# load trainset
fh = open("train.pkl", "r")
data = cPickle.load(fh)
fh.close

# train RSM
start_time = time.time()
r = data.shape[0]/batchsize*batchsize
layer = rsm_numpy.RSM()
result = layer.train(data[0:r], hiddens, epochs, lr=learning_rate, btsz=batchsize)
print "Time: " + str(time.time() - start_time)

# save results
fh = open("rsm_result.pkl", "w")
cPickle.dump(result, "rsm_result")
fh.close()
