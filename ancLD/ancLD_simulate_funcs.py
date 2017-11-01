
# ## Reworked the two-locus simualtion code to be a bit more clear
# ### still need to re-incorporate it into a broad strategy that allows evalaution of results across many simulation runs

import collections

import numpy as np
import pandas as pd
import scipy.stats


import LDadmix_v8_funcs as LDadmix


# generate the Q array
def generate_Q_beta(a, b, n):
	"""use the beta distribution to generate a vector of Q values"""
	Q = scipy.stats.beta.rvs(a, b, size = n)
	#Q = np.vstack([Q, 1-Q]) # if i want a matrix
	return(Q)


def get_hap_freqs_fromLD(true_r2, true_p1, true_p2):
	"""Given parametric r2 and allele freqeuncies, calculate the haplotype freqeuncies using their relationship with D^2"""
	pA = true_p1
	pa = 1.0 - pA
	pB = true_p2
	pb = 1.0 - pB

	D = np.sqrt(true_r2*pA*pa*pB*pb)
	assert(D <= 1)
	assert(D >= -1)

	pAB = pA*pB + D
	pAb = pA*pb - D
	paB = pa*pB - D
	pab = pa*pb + D

	assert(pAB <= 1)
	assert(pAB >= 0)
	assert(pAb <= 1)
	assert(pAb >= 0)
	assert(paB <= 1)
	assert(paB >= 0)
	assert(pab <= 1)
	assert(pab >= 0)
	return(np.array([pab, paB, pAb, pAB]))


def sample_haplotypes(n, pab, paB, pAb, pAB):
	"""given 4 haplotype freqeuncies, return n haplotypes sampled from those freqeuenices """
	assert (np.allclose(1.0, pab+ paB + pAb + pAB)) # freqeuncies must add to 1
	choices = np.random.choice(a = 4, size = n, replace = True, p = [pab, paB, pAb, pAB])
	haps = np.zeros((n, 2))
	haps[choices == 3] = np.array([1,1])
	haps[choices == 2] = np.array([1,0])
	haps[choices == 1] = np.array([0,1])
	haps[choices == 0] = np.array([0,0])
	return(haps.astype('int64'))


def get_hap_counts(hap_pairs):
	"""given an array of haplotypes, return counts of the four haplotypes"""
	hap = 2*hap_pairs[:,0] + hap_pairs[:,1]
	#haps, counts = np.unique(hap, return_counts=True)
	hap0 = (hap==0).sum()
	hap1 = (hap==1).sum()
	hap2 = (hap==2).sum()
	hap3 = (hap==3).sum()
	return(np.array([hap0, hap1, hap2, hap3]))


def get_hap_freqs(hap_pairs):
	"""given an array of haplotypes, return frequencies of the four haplotypes"""

	#re = map(get_hap_counts, hap_pairs)
	re = get_hap_counts(hap_pairs)
	re = np.array(re).astype('float') #ensures hap_freqs are floats
	hap_freqs = re / re.sum()#[:, np.newaxis]
	return(hap_freqs)


def get_LD(p1, p2, H4):
	pA = p1
	pB = p2
	pAB= H4
	# D
	D = pAB - pA*pB
	pa = 1.0 - pA
	pb = 1.0 - pB
	# Dprime
	pApB = pA*pB
	pApb = pA*pb
	papB = pa*pB
	papb = pa*pb
	A = np.minimum(pApb, papB) # Dmax when D is positive
	B = np.minimum(pApB, papb) # Dmax when D is negative
	Dmax = np.where(D >= 0, A, B)
	Dprime = D/Dmax
	# r^2
	try:
		r2 = (D**2)/(pA*pB*pa*pb)
	except ZeroDivisionError:
		r2 = np.nan
	return(r2, D, Dprime)
