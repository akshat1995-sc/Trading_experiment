import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import minimize
from scipy.linalg import toeplitz
from scipy import signal


												#Creating the dataset
#Attribute of noise
drift_rt=0.15
ini_val=100
volat=0.3
del_t=0.0192																#Discrete time steps (including initial state)
end_time=100*del_t
num_exp=int(1)																#Number of experiments
time=np.arange(0,end_time,del_t)											#Time vector
x_0=ini_val*np.ones([num_exp,1])												#Initial State
xt=np.c_[x_0,np.zeros([num_exp,time.shape[0]-1])]							#Initiating dynamic storage matrix

#Stochastic simulation
for i in range(0,num_exp):
	vt=np.random.normal(0,1,size=time.shape)								#Producing random variable
	for j in range(1,time.shape[0]):
		xt[i,j]=xt[i,j-1]*(1+drift_rt*del_t+volat*vt[j]*np.sqrt(del_t))		#x(t) = v(t)*srqt(t)



												#Wiener-Hopf algorithm
N = len(xt[0,:]) 												# number of data points per signal
x = np.arange(0, N)  											# point indices
T = 50
v_2=np.random.normal(0,1,size=time.shape)						#Producing random variable


y = xt[0,:]
r_v2 = np.zeros((T))
r_v1v2 = np.zeros((T))

for i in range(0, T - 1):
    for j in range(i + 1, N):
        r_v2[i] +=  1 / (N - i) * v_2[j] * v_2[j - i] 
        r_v1v2[i] += 1 / (N - i) * y[j] * v_2[j - i]          
    

R_v2 = toeplitz(r_v2, r_v2)
R_v2_inv = np.linalg.inv(R_v2)

w_hat = np.matmul(R_v2_inv, r_v1v2)
vt_approx = signal.lfilter(w_hat, 1, v_2)


												#Optimization algorithm

def objective(x):
	dat=np.zeros(time.shape[0])
	dat[0]=ini_val
	for i in range(1,time.shape[0]):
		dat[i]=dat[i-1]*(1+x[0]*del_t+x[1]*vt_approx[i]*np.sqrt(del_t))
	err=np.dot(xt.reshape(-1,)-dat,xt.reshape(-1,)-dat)
	return err

def constraint1(x):
    return x[1]

# initial guesses
n = 2
x0 = np.zeros(n)
x0[0] = 0.1
x0[1] = 0.1


# show initial objective
print('Initial SSE Objective: ' + str(objective(x0)))

# optimize
cons = {'type': 'ineq', 'fun': constraint1}
solution = minimize(objective,x0,method='SLSQP',\
                    constraints=cons)
x = solution.x

# show final objective
print('Final SSE Objective: ' + str(objective(x)))

# print solution
print('Solution')
print('x1 = ' + str(x[0]))
print('x2 = ' + str(x[1]))

x_new=np.c_[x_0,np.zeros([num_exp,time.shape[0]-1])]								#Initiating dynamic storage matrix

#Stochastic simulation
for i in range(0,num_exp):
	for j in range(1,time.shape[0]):
		x_new[i,j]=x_new[i,j-1]*(1+x[0]*del_t+x[1]*vt_approx[j]*np.sqrt(del_t))			#x(t) = v(t)*srqt(t)

# plt.plot(xt[0,:],color='red')
# plt.plot(x_new[0,:],color='blue')
plt.plot(vt)
plt.plot(vt_approx)
plt.show()