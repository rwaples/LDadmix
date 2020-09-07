from __future__ import absolute_import, division, print_function
from builtins import (ascii, bytes, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      str, super, zip)# http://python-future.org/imports.html

import itertools
import ctypes
import numpy as np

# my settings
_numba_available = True
try:
	from numba import jit
	print("------------------")
	print("Numba import successful")
	print("------------------\n")
except ImportError:
	_numba_available = False
	print("------------------")
	print("Import of numba failed, analysis will run significantly slower")
	print("------------------\n")

# decorator depending on numba, if numba is found, will decorate with: jit(...)
def optional_numba_decorator(func):
	if _numba_available:
		# not sure how to specify nogil: False or True?
		# maybe break into one for each
		return(jit(nopython=True, nogil=False, cache=False)(func))
	else:
		return(func)

# pos can be a vector of float or int posistions, as can the maxdist
# dont set to cache
@jit(nopython=True, nogil=False, cache=False)
def find_idxpairs_maxdist_generator(pos, maxdist):
	"""iterator, yields indexes for all pairs of loci within maxdist,
	needs to be grouped into pairs
	pos : positions of loci
	maxdist : maximum distance apart"""
	siteix = 0
	for sitepos in pos:
		if maxdist == 0:
			leftix = 0
		else:
			# index of the leftmost locus position within maxdist
			leftix = np.searchsorted(pos, sitepos - maxdist)
		for altix in range(leftix, siteix):
			yield altix
			yield siteix
		siteix += 1

@jit(nopython=True, nogil=True, cache=True)
def get_rand_hap_freqs(n=2, seed = 0):
	"""returns an (n,4) dimensioned numpy array of random haplotype frequencies,
	where n is the number of pops"""
	if seed == 0:
		np.random.seed(seed)
	else:
		pass
	H = np.zeros((n, 4), dtype = np.float64)
	for i in range(n):
		H[i] = np.diff(np.concatenate((np.array([0]), np.sort(np.random.rand(3)), np.array([1.0]))))
	return(H)


@jit(nopython=True, nogil=True, cache=True)
def map2domain(H, minfreq):
	"""ensures that frequencies haver get too close to 0 or 1
	limits determied """
	maxfreq = 1-minfreq
	a,b = H.shape
	asum = np.zeros(a)
	for i in range(a):
		for j in range(b):
			if H[i,j]<minfreq:
				H[i,j]=minfreq
			if H[i,j]>maxfreq:
				H[i,j]=maxfreq
			asum[i] += H[i,j]
	for i in range(a):
		for j in range(b):
			H[i,j] = H[i,j]/asum[i]
	return(H)


@jit(nopython=True, nogil=False, cache=True)
def get_sumstats_from_haplotype_freqs(H):
	"""given a set of four haplotype frequencies x populations,
	returns r^2, D, Dprime, and allele frequencies in each pop"""
	# 00, 01, 10, 11
	pA = H[:,2] + H[:,3]
	pB = H[:,1] + H[:,3]
	pAB= H[:,3]
	D = pAB - pA*pB
	# expected freqs
	pa = 1.0 - pA
	pb = 1.0 - pB
	pApB = pA*pB
	pApb = pA*pb
	papB = pa*pB
	papb = pa*pb
	A = np.minimum(pApb, papB) # Dmax when D is positive
	B = np.minimum(pApB, papb) # Dmax when D is negative
	Dmax = np.where(D >= 0, A, B)
	Dprime = np.abs(D/Dmax) # now an absolute value
	r2 = (D**2)/(pA*pB*pa*pb)
	return(r2, D, Dprime, pA, pB)

@jit(nopython=True, nogil=False, cache=True)
def flag_maf(H, maf):
	upper_maf = 1-maf
	p1 = H[:, 2] + H[:, 3]
	p2 = H[:, 1] + H[:, 3]
	flags = (p1<maf) + (p1>upper_maf) + (p2<maf) + (p2>upper_maf)
	return(flags)


@jit(nopython=True, nogil=False, cache=True)
def get_LL_numba(Q, H, code):
	"""returns the loglikelihood of the genotype data given Q and H
	Q = admixture fractions
	H = estimates of source-specific haplotype frequencies"""

	ind_hap_freqs = np.dot(Q, H)
	LL = 0.0
	# had to index the tuples returned by np.where
	LL += np.log(	ind_hap_freqs[np.where(code == 0)[0], 0] * ind_hap_freqs[np.where(code == 0)[0], 0]).sum()
	LL += np.log(2 * ind_hap_freqs[np.where(code == 1)[0], 0] * ind_hap_freqs[np.where(code == 1)[0], 2]).sum()
	LL += np.log(	ind_hap_freqs[np.where(code == 2)[0], 2] * ind_hap_freqs[np.where(code == 2)[0], 2]).sum()
	LL += np.log(2 * ind_hap_freqs[np.where(code == 3)[0], 0] * ind_hap_freqs[np.where(code == 3)[0], 1]).sum()
	LL += np.log(2 * ind_hap_freqs[np.where(code == 4)[0], 0] * ind_hap_freqs[np.where(code == 4)[0], 3]
			   + 2 * ind_hap_freqs[np.where(code == 4)[0], 1] * ind_hap_freqs[np.where(code == 4)[0], 2]).sum()
	LL += np.log(2 * ind_hap_freqs[np.where(code == 5)[0], 2] * ind_hap_freqs[np.where(code == 5)[0], 3]).sum()
	LL += np.log(	ind_hap_freqs[np.where(code == 6)[0], 1] * ind_hap_freqs[np.where(code == 6)[0], 1]).sum()
	LL += np.log(2 * ind_hap_freqs[np.where(code == 7)[0], 1] * ind_hap_freqs[np.where(code == 7)[0], 3]).sum()
	LL += np.log(	ind_hap_freqs[np.where(code == 8)[0], 3] * ind_hap_freqs[np.where(code == 8)[0], 3]).sum()
	return(LL)

def get_geno_codes(genos):
	"""maps each pair of non-missing genotypes into an integer from 0 to 8
	pairs with missing genotypes will map above 8"""
	return(genos[0] + 3*genos[1])
