import pylab as py
import scipy
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Randomly generated data: Training/Testing -----------------------------------
# -----------------------------------------------------------------------------

np.random.seed(282679734) # set random seed

Ns =  10000 # Number of positives
Nb = 50000 # Number of negatives

# Observables for positives
s1 = np.random.normal(1.2,1.0,Ns) # normal distribution
s2 = np.random.gamma(1.5,2,Ns) # gamma distribution
s3 = np.random.beta(1.5,1.0,Ns) # beta distribution
s4 = np.random.exponential(0.5,Ns) # exponential distribution
s5 = np.hstack((np.ones(0.90*Ns),np.zeros(0.1*Ns))) # binary
s6 = np.hstack((np.zeros(0.35*Ns),1*np.ones(0.15*Ns),2*np.ones(0.1*Ns),3*np.ones(0.4*Ns))) # quadtrary 
s7 = (np.random.normal(1.7,0.2,Ns),np.random.normal(1,0.5,Ns)) # 2D normal distribution

# Observables for negatives
b1 = np.random.normal(0,1.0,Nb)
b2 = np.random.gamma(1.5,1.0,Nb)  
b3 = np.random.beta(1.5,2.0,Nb)
b4 = np.random.exponential(0.3,Nb) 
b5 = np.hstack((np.ones(0.30*Nb),np.zeros(0.70*Nb)))   
b6 = np.hstack((np.zeros(0.2*Nb),1*np.ones(0.25*Nb),2*np.ones(0.2*Nb),3*np.ones(0.35*Nb))) # quadtrary np.hstack() # binary
b7 = (np.random.normal(1.0,0.8,Nb),np.random.normal(0.5,0.1,Nb))



# Distributions for inputs for training and test data
# Number of histogram bins
Nbins = 50

plt.figure(1, figsize=(12,12)).patch.set_facecolor('white')
plt.subplot(321)
plt.hist([s1,b1],Nbins,histtype='step')
plt.title('Normal Dist.')
plt.subplot(322)
plt.hist([s2,b2],Nbins,histtype='step',label=['Positive','Negatives'])
plt.title('Gamma Dist.')
plt.xlim(0,10)
plt.legend()
plt.subplot(323)
plt.hist([s3,b3],Nbins,histtype='step')
plt.title('Beta Dist.')
plt.subplot(324)
plt.hist([s4,b4],Nbins,histtype='step')
plt.title('Exponential Dist.')    
plt.xlim(0,2)
plt.subplot(325)
plt.hist([s5,b5],2,histtype='step')
plt.title('Binary Dist.') 
plt.ylim(0,37000)
plt.subplot(326)
plt.hist([s6,b6],4,histtype='step')
plt.title('Quadrary Dist.') 
#plt.savefig('/home/rmr/randDidst.png', bbox_inches=0)

cs12 = scipy.stats.pearsonr(s1,s2)[0]
cs13 = scipy.stats.pearsonr(s1,s3)[0]
cs14 = scipy.stats.pearsonr(s1,s4)[0]
cs15 = scipy.stats.pearsonr(s1,s5)[0]
cs16 = scipy.stats.pearsonr(s1,s6)[0]

cb12 = scipy.stats.pearsonr(b1,b2)[0]
cb13 = scipy.stats.pearsonr(b1,b3)[0]
cb14 = scipy.stats.pearsonr(b1,b4)[0]
cb15 = scipy.stats.pearsonr(b1,b5)[0]
cb16 = scipy.stats.pearsonr(b1,b6)[0]

cs23 = scipy.stats.pearsonr(s2,s3)[0]
cs24 = scipy.stats.pearsonr(s2,s4)[0]
cs25 = scipy.stats.pearsonr(s2,s5)[0]
cs26 = scipy.stats.pearsonr(s2,s6)[0]

cb23 = scipy.stats.pearsonr(b2,b3)[0]
cb24 = scipy.stats.pearsonr(b2,b4)[0]
cb25 = scipy.stats.pearsonr(b3,b5)[0]
cb26 = scipy.stats.pearsonr(b3,b6)[0]

cs34 = scipy.stats.pearsonr(s3,s4)[0]
cs35 = scipy.stats.pearsonr(s3,s5)[0]
cs36 = scipy.stats.pearsonr(s3,s6)[0]

cb34 = scipy.stats.pearsonr(b3,b4)[0]
cb35 = scipy.stats.pearsonr(b3,b5)[0]
cb36 = scipy.stats.pearsonr(b3,b6)[0]

cs45 = scipy.stats.pearsonr(s4,s5)[0]
cs46 = scipy.stats.pearsonr(s4,s6)[0]

cb45 = scipy.stats.pearsonr(s4,s5)[0]
cb46 = scipy.stats.pearsonr(b4,b6)[0]

cs56 = scipy.stats.pearsonr(s5,s6)[0]
cb56 = scipy.stats.pearsonr(b5,b6)[0]

fig = plt.figure(2, figsize=(12,12)).patch.set_facecolor('white')
ax = plt.subplot(221)
ax.scatter(s1, s2,s=10, c='r', marker="o", alpha=1, label='C_Pos = %0.4f' % cs12)
ax.scatter(b1, b2, s=10,c='b', marker="o", alpha=0.1, label='C_Neg = %0.4f' % cb12)
plt.title('Normal - Gamma')
plt.legend(loc='upper right')
ax = plt.subplot(222)
ax.scatter(s1, s3,s=100, c='r', marker="o", alpha=1, label = 'Positive, C_Pos = %0.4f' % cs13)
ax.scatter(b1, b3, s=100,c='b', marker="o", alpha=0.1, label = 'Negative, C_N = %0.4f' % cb13)
plt.title('Normal - Beta')
plt.legend(loc='upper right')
ax = plt.subplot(223)
ax.scatter(s1, s4,s=100, c='r', marker="o", alpha=1, label='C_Pos = %0.4f' % cs14)
ax.scatter(b1, b4, s=100,c='b', marker="o", alpha=0.1, label='C_Neg = %0.4f' % cb14)
plt.title('Normal - Exponential')
plt.legend(loc='upper right')
ax = plt.subplot(224)
ax.scatter(s1, s5,s=100, c='r', marker="o", alpha=1, label='C_Pos = %0.4f' % cs15)
ax.scatter(b1, b5, s=100,c='b', marker="o", alpha=0.1, label='C_Neg = %0.4f' % cb15)
plt.title('Normal - Binary')
plt.legend(loc='upper right')

fig = plt.figure(3, figsize=(12,12)).patch.set_facecolor('white')
ax = plt.subplot(221)
ax.scatter(s1, s6,s=100, c='r', marker="o", alpha=1, label='C_Pos = %0.4f' % cs16)
ax.scatter(b1, b6, s=100,c='b', marker="o", alpha=0.1, label='C_Neg = %0.4f' % cb16)
plt.title('Normal - Quadrary')
plt.legend(loc='upper right')
ax = plt.subplot(222)
ax.scatter(s2, s3,s=100, c='r', marker="o", alpha=1, label ='Positive, C_Pos = %0.4f' % cs23)
ax.scatter(b2, b3, s=100,c='b', marker="o", alpha=0.1, label = 'Negative C_Neg = %0.4f' % cb23)
plt.title('Gamma - Beta')
plt.legend(loc='upper right')
ax = plt.subplot(223)
ax.scatter(s2, s4,s=100, c='r', marker="o", alpha=1, label='C_Pos = %0.4f' % cs24)
ax.scatter(b2, b4, s=100,c='b', marker="o", alpha=0.1, label='C_Neg = %0.4f' % cb24)
plt.title('Gamma - Exponential')
plt.legend(loc='upper right')
ax = plt.subplot(224)
ax.scatter(s2, s5,s=100, c='r', marker="o", alpha=1, label='C_Pos = %0.4f' % cs25)
ax.scatter(b2, b5, s=100,c='b', marker="o", alpha=0.1, label='C_Neg = %0.4f' % cb25)
plt.title('Gamma - Binary')
plt.legend(loc='upper right')

fig = plt.figure(4, figsize=(12,12)).patch.set_facecolor('white')
ax = plt.subplot(221)
ax.scatter(s2, s6,s=100, c='r', marker="o", alpha=1, label='C_Pos = %0.4f' % cs26)
ax.scatter(b2, b6, s=100,c='b', marker="o", alpha=0.1, label='C_Neg = %0.4f' % cb26)
plt.title('Gamma - Quadrary')
plt.legend(loc='upper right')
ax = plt.subplot(222)
ax.scatter(s3, s4,s=100, c='r', marker="o", alpha=1, label = 'Positive, C_Pos = %0.4f' % cs34)
ax.scatter(b3, b4, s=100,c='b', marker="o", alpha=0.1, label = 'Negative, C_Neg = %0.4f' % cb34)
plt.title('Beta - Exponential')
plt.legend(loc='upper right')
ax = plt.subplot(223)
ax.scatter(s3, s5,s=100, c='r', marker="o", alpha=1, label='C_Pos = %0.4f' % cs35)
ax.scatter(b3, b5, s=100,c='b', marker="o", alpha=0.1, label='C_Neg = %0.4f' % cb35)
plt.title('Beta - Binary')
plt.legend(loc='upper right')
ax = plt.subplot(224)
ax.scatter(s3, s6,s=100, c='r', marker="o", alpha=1, label='C_Pos = %0.4f' % cs36)
ax.scatter(b3, b6, s=100,c='b', marker="o", alpha=0.1, label='C_Neg = %0.4f' % cb36)
plt.title('Beta - Quadrary')
plt.legend(loc='upper right')

fig = plt.figure(5, figsize=(12,12)).patch.set_facecolor('white')
ax = plt.subplot(131)
ax.scatter(s4, s5,s=100, c='r', marker="o", alpha=1, label='C_Pos = %0.4f' % cs45)
ax.scatter(b4, b5, s=100,c='b', marker="o", alpha=0.1, label='C_Neg = %0.4f' % cb45)
plt.title('Exponential - Binary')
ax = plt.subplot(132)
plt.legend(loc='upper right')
ax.scatter(s4, s6,s=100, c='r', marker="o", alpha=1, label='C_Pos = %0.4f' % cs46)
ax.scatter(b4, b6, s=100,c='b', marker="o", alpha=0.1, label='C_Neg = %0.4f' % cb46)
plt.title('Exponential - Quadrary')
ax = plt.subplot(133)
ax.scatter(s5, s6,s=100, c='r', marker="o", alpha=1, label = 'Positive, C_Pos = %0.4f' % cs56)
ax.scatter(b5, b6, s=100, c='b', marker="o", alpha=0.1, label = 'Negative, C_Neg = %0.4f' % cb56)
plt.title('Binary - Quadrary')
plt.legend(loc='upper right')

plt.show()