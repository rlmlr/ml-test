import numpy as np
import pandas as pd
from random import sample
from sklearn import preprocessing
# -----------------------------------------------------------------------------
# Randomly generated data: Training/Testing -----------------------------------
# -----------------------------------------------------------------------------

def rData(tn, seed, N0, N1, ep): 

	np.random.seed(seed) # set random seed

	Ns = N1 # Number of positives
	Nb = N0 # Number of negatives

	# Observables for positives
	s1 = np.random.normal(1.2-ep*1.2,1.0,Ns) # normal distribution
	s2 = np.random.gamma(1.5-ep*1.5,2,Ns) # gamma distribution
	s3 = np.random.beta(1.5-ep*1.5,1.0,Ns) # beta distribution
	s4 = np.random.exponential(0.5-ep*0.5,Ns) # exponential distribution
	s5 = np.hstack((np.ones(0.90*Ns),np.zeros(0.1*Ns))) # binary
	s6 = np.hstack((np.zeros(0.35*Ns),1*np.ones(0.15*Ns),2*np.ones(0.1*Ns),3*np.ones(0.4*Ns))) # quadtrary 
	s7 = np.ones(Ns) # Positive target value  = 1
 
	# Observables for negatives
	b1 = np.random.normal(0,1.0,Nb)
	b2 = np.random.gamma(1.5,1.0,Nb)
	b3 = np.random.beta(1.5,2.0,Nb)
	b4 = np.random.exponential(0.3,Nb)
	b5 = np.hstack((np.ones(0.30*Nb),np.zeros(0.70*Nb)))
	b6 = np.hstack((np.zeros(0.2*Nb),1*np.ones(0.25*Nb),2*np.ones(0.2*Nb),3*np.ones(0.35*Nb))) # quadtrary np.hstack() # binary 
	b7 = np.zeros(Nb) # Negative target value  = 0

	# Stack data horizontally
	x1 = np.hstack((s1,b1))
	x2 = np.hstack((s2,b2))
	x3 = np.hstack((s3,b3))
	x4 = np.hstack((s4,b4))
	x5 = np.hstack((s5,b5))
	x6 = np.hstack((s6,b6))

	# Stack target values horizontally and convert to integer
	targets = np.hstack((s7,b7)).astype(int)

	# Data Frame ------------------------------------------------------------------

	# Dictionaries containing labels as keys and data as values
	d = {'Norm' : x1,'Gamma' : x2, 'Beta' : x3, 'Exp' : x4, 'Binary': x5, 'Quad': x6}

	# Create data frame from dictionary d
	df = pd.DataFrame(d)

	# Add boolean variable to separate training and testing data (%75 train, 25% test)
	df['isTrain'] = np.random.uniform(0, 1, len(df)) <= 0.75
	# Assign a name Negatives or Positives to target value
	df['Type'] = pd.Factor(targets, tn)
	# Add target values
	df['Targets'] = targets

	# Prep Data for model input ---------------------------------------------------

	# Create randomized index to randomize data frame by rows
	rindex =  np.array(sample(xrange(len(df)), len(df)))

	# Re-index data frame
	RDF = df.ix[rindex]

	return RDF
