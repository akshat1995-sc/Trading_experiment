import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import minimize
from scipy.linalg import toeplitz
from scipy import signal


												#Creating the dataset
#Attribute of noise
def brown_motion(dfrt,volat,del_t,ini_val,time,vt):
	x_0=ini_val*np.ones([num_exp,1])										#Initial State
	xt=np.c_[x_0,np.zeros([num_exp,time.shape[0]-1])]						#Initiating dynamic storage matrix

	#Stochastic simulation
	for i in range(0,num_exp):
		for j in range(1,time.shape[0]):
			xt[i,j]=xt[i,j-1]*(1+drift_rt*del_t+volat*vt[j]*np.sqrt(del_t))	#x(t) = v(t)*srqt(t)
	return(xt[0,:])



												#Optimization algorithm
drift_rt=0.15
ini_val=100
volat=0.3
del_t=0.0192
end_time=100*del_t
num_exp=int(1)																#Number of experiments
time=np.arange(0,end_time,del_t)											#Time vector
vt=np.random.normal(0,1,size=time.shape)									#Producing random variable
xt=brown_motion(drift_rt,volat,del_t,ini_val,time,vt)						#Discrete time steps (including initial state)

def objective(x):
	ret=[((xt[i+1]-xt[i])/xt[i]) for i in range(len(xt)-1)]
	err=np.empty(shape=(1,))
	for i in range(0,time.shape[0]-1):
		err=err+ret[i]**2+x[0]**2+x[1]**2-2*x[0]*ret[i]
	return err

def constraint1(x):
    return x[1]

def constraint2(x):
    return x[0]+1

def constraint3(x):
    return -x[0]+1

# initial guesses
n = 2
x0 = np.zeros(n)
x0[0] = 0.001
x0[1] = 0.002


# show initial objective
print('Initial SSE Objective: ' + str(objective(x0)))

# optimize
cons = [{'type': 'ineq', 'fun': constraint1},\
		{'type': 'ineq', 'fun': constraint2},\
		{'type': 'ineq', 'fun': constraint3}]

# vt=np.random.normal(0,1,size=time.shape)									#Producing random variable

def eval_mu(x0,ini_val,vt=vt):
	solution = minimize(objective,x0,method='SLSQP')#,\
	                    # constraints=cons)
	x = solution.x

	# show final objective
	print('Final SSE Objective: ' + str(objective(x)))

	# print solution
	print('Solution')
	print('x1 = ' + str(x[0]))
	print('x2 = ' + str(x[1]))
	return (x[0]/del_t)
