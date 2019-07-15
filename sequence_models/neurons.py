import numpy as np 
activate = lambda x:np.tanh(x)
dActivate = lambda x,dx:(1 - x**2)*dx

def sigmoid(x, derivative=False):
    if derivative:
    	return x * (1 - x)# Derivative Equation
    return 1 / (1 + np.exp(-x))


class RNN:
	
	def __init__(self):
		pass
	def forward(self,x,name):
		self.x = x
		self.hx = activate(self.x)
		self.name = name
		return self.hx
	def backward(self,hs,db):
		self.x = hs;
		self.dx = db
		self.dhx = dActivate(self.x,self.dx)
		return self.dhx

# class LSTM:

# 	def __init__(self):
# 		pass

# 	def forward():
# 		ft = sigmoid(self.wf*self.s + self.wf*x)#forget gate
# 		it = sigmoid(self.wi*self.s + self.wi*x)# input gate
		
