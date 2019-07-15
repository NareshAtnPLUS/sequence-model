''''Utility functions for Sequence Models.'''
import numpy as np

softmax = lambda y:np.exp(y)/np.sum(np.exp(y))
one_hot_encoder=lambda vocab_size,vector:np.eye(vocab_size)[vector]
def adagrad(weighs,delta,mem,alpha=1e-1):
	# AdaGrad Update 
	for param,dparam,mem in zip(weighs,delta,mem):
		mem += dparam * dparam

		param += -alpha * dparam / np.sqrt(mem + 1e-8)
	return weighs
def sigmoid(x,derivative=False):
	if derivative:
		return (1 - x)*x
	return 1/(1+np.exp(-(x)))