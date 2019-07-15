"""Minimal Character level Vanilla RNN.Written in Glosys Research by nPLUS"""
import numpy as np 
from sequence_models.utils import *
from sequence_models.neurons import RNN

data = open('../data/word.txt').read()

chars = list(set(data))
data_size, vocab_size = len(data),len(chars)

char_to_ix = {'h':0,'e':1,'l':2,'o':3}
ix_to_char = {0:'h',1:'e',2:'l',3:'o'}

input_vector = [char_to_ix[ch] for ch in data[:len(data)-1]]
target = one_hot_encoder(vocab_size,char_to_ix['o']).reshape(vocab_size,1)
print(input_vector,char_to_ix['o'])
one_hot = one_hot_encoder(vocab_size,input_vector)

print('target',target.shape,one_hot.shape)

def vanilla_rnn(Wxh,Whh,bh,Why,alpha=1e-1):
	hs,ys,ps,cell={},{},{},{}
	dWxh, dWhh, dWhy,dbh = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why),np.zeros_like(bh)
	

	hs[-1] = np.zeros((3,1))
	#print(one_hot)
	for t in range(len(one_hot)):
		one = one_hot[t].reshape(vocab_size,1)
		#print(f'''\n{ix_to_char[np.argmax(one)]} \n{one}\nwxh \n{Wxh}''')
		h=Wxh.dot(one)
		hh = hs[t-1].dot(Whh) + bh
		cell[t] = RNN()
		hs[t] = (cell[t]).forward(hh+h,name=str(t))
		ys[t] = Why.dot(hs[t])
		ps[t] = softmax(ys[t])
		''' Uncomment this line when it's required to investigate nodes at each timestep'''
		"""print(f'''\n\ntimestep t => {t+1}\n\nhs[t] \n{hs[t]}\n\nhs[t-1] \n{hs[t-1]}
			\n\nys[t] \n{ys[t]}\n\nps[t] \n{ps[t]}\nps[t][2]\n{ps[t][3]}''')"""

	loss = -np.log(ps[t][3])
	dy = np.copy(ps[t])
	#print(,t,(target[t]))
	dy[np.argmax(target)] -= 1
	#print('dy',dy,sep='\n')
	dWhy += dy.dot(hs[t].T)
	dh = Why.T.dot(dy)

	for t in reversed(range(len(one_hot))):
		xs = one_hot[t].reshape(vocab_size,1)
		db_act = (cell[t]).backward(hs[t],dh)
		dWxh += db_act.dot(xs.T)
		dWhh += db_act.T.dot(hs[t-1])
	# print(f'dh\n{dh}\ndb_act\n{db_act}')
	# print(f'Deltas of the  Learning Parameters\ndWxh\n{dWxh}\ndWhh\n{dWhh}\ndWhy\n{dWhy}')

	
	return dWxh, dWhh, dWhy,loss

def vanilla_rnn_test(Wxh,Whh,bh,Why):
	hs,ys,ps={},{},{}
	dWxh, dWhh, dWhy,dbh = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why),np.zeros_like(bh)
	

	hs[-1] = np.zeros((3,1))
	#print(one_hot)
	for t in range(len(one_hot)):
		one = one_hot[t].reshape(vocab_size,1)
		#print(f'''\n{ix_to_char[np.argmax(one)]} \n{one}\nwxh \n{Wxh}''')
		h=Wxh.dot(one)
		hh = hs[t-1].dot(Whh) + bh
		hs[t] = np.tanh(hh+h)
		ys[t] = Why.dot(hs[t])
		ps[t] = softmax(ys[t])

		# print(f'''\n\ntimestep t => {t+1}\n\nhs[t] \n{hs[t]}\n\nhs[t-1] \n{hs[t-1]}
		# 	\n\nys[t] \n{ys[t]}\n\nps[t] \n{ps[t]}\nps[t][2]\n{ps[t][3]}''')
	loss = -np.log(ps[t][3])
	return ix_to_char[np.argmax(ps[t])]
