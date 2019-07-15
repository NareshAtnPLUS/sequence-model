import numpy as np 
import matplotlib.pyplot as plt 
from sequence_models.vanilla_rnn import vanilla_rnn,vanilla_rnn_test,one_hot
from sequence_models.utils import adagrad 

def run():
	Wxh,Whh,bh,Why = (np.array([[0.287027,0.84606,0.572392,0.486813],
	[0.902874,0.871522,0.691079,0.18998],
	[0.537524,0.09234,0.55815,0.491528]]),
	np.array([[0.427043]]),np.array([[0.567001]]),
	np.array([
	[0.37168, 0.974829459, 0.830034886],
	[0.39141, 0.282585823,0.659835709],
	[0.64985, 0.09821557, 0.3342870884],
	[0.91266, 0.32581642, 0.144630018]]))
	mWxh, mWhh, mWhy, mbh = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why), np.zeros_like(bh)
	loss_list = []
	for i in range(100):
		dWxh, dWhh, dWhy,loss = vanilla_rnn(Wxh,Whh,bh,Why)
		loss_list.append(loss)
		weighs,delta,mem = [Wxh,Whh,Why],[dWxh, dWhh, dWhy],[mWxh, mWhh, mWhy]
		if i % 25 == 0:
			print(f'At iteration:{i}\nLearning Parameters Before AdaGrad update \nWxh\n{Wxh}\nWhh\n{Whh}\nWhy\n{Why}')		
		weighs = adagrad(weighs,delta,mem)
		if i % 25 == 0:
			print(f'At iteration:{i}\nLearning Parameters After AdaGrad update \nWxh\n{Wxh}\nWhh\n{Whh}\nWhy\n{Why}')				

	ps = vanilla_rnn_test(Wxh,Whh,bh,Why)
	print("next predicted Character for the given corpus(hell):",ps)

	plt.plot(loss_list)
	plt.show()