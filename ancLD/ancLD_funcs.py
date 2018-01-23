import multiprocessing
import itertools
import ctypes

import numpy as np
from plinkio import plinkfile
#import pandas as pd

NUMBA_DISABLE_JIT = 0
NUMBA = False
try:
	from numba import jit
	NUMBA = True
	print("------------------")
	print("Numba import successful")
	print("------------------\n")
except ImportError:
	print("------------------")
	print("Import of numba failed, analysis will run significantly slower")
	print("------------------\n")
# decorator depending on numba, if numba is found, will decorate with: jit(nopython=True)
def optional_numba_decorator(func):
	if NUMBA:
		#return(jit(nopython=True)(func))
		return(jit(nopython=True, nogil=True, cache=False)(func))
	else:
		return(func)


def load_plinkfile(basepath):
	plink_file = plinkfile.open(basepath)
	sample_list = plink_file.get_samples()
	locus_list = plink_file.get_loci()
	my_array = np.zeros((len(plink_file.get_loci( )), len(plink_file.get_samples( ))))
	for i, el in enumerate(plink_file):
		my_array[i] = el
	# look for missing data
	has_missing = False
	if 3 in np.unique(my_array):
		has_missing = True
		# replace missing values with nines - this will cause them to never match during genotype code checking
		my_array[my_array == 3] = 9

	my_array = my_array.astype('i1')
	return(sample_list, locus_list, my_array, has_missing)


def find_idxpairs_maxdist_generator(pos, maxdist):
	"""iterator, yields tuples of indexes for all pairs of loci within maxdist
	pos : positions of loci
	maxdist : maximum distance apart"""
	for siteix, sitepos in enumerate(pos):
		if np.allclose(maxdist, 0):
			leftix = 0
		else:
			leftix = np.searchsorted(pos, sitepos - maxdist) # index of the leftmost locus position with maxdist
		for altix in xrange(leftix, siteix):
			#yield ((altix, siteix))
			yield altix
			yield siteix


@optional_numba_decorator
def get_rand_hap_freqs(n=2, seed = None):
	"""returns an (n,4) dimensioned numpy array of random haplotype frequencies,
	where n is the number of pops"""
	if seed:
		np.random.seed(seed)
	else:
		pass
	H = np.zeros((n, 4))
	for i in xrange(n):
		H[i] = np.diff(np.concatenate((np.array([0]), np.sort(np.random.rand(3)), np.array([1.0]))))
	return(H)


@optional_numba_decorator
def get_geno_codes(genos):
	"""maps each pair of non-missing genotypes into an integer from 0 to 8
	pairs with missing genotypes will map above 8"""
	return(genos[0] + 3*genos[1])


@optional_numba_decorator
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


# try G as a global
#G = np.array([0,3,1,4, 3,6,4,7, 1,4,2,5, 4,7,5,8])

@optional_numba_decorator
def do_multiEM(inputs):
	""""""
	H, Q, code, max_iter, tol = inputs # unpack the input

	n_ind = Q.shape[0]
	n_pops = Q.shape[1]
	G = np.array([0,3,1,4, 3,6,4,7, 1,4,2,5, 4,7,5,8]) # which combinations of haplotypes produce which genotypes

	old_LL = get_LL_numba(Q = Q, H = H , code = code)

	# start iteration here
	for i in xrange(1, max_iter+1):
		norm = np.zeros(n_ind) # maybe just change this to a matrix of ones - then set to zero if data is found to be non-missing
		isum = np.zeros((n_ind, n_pops, 4)) # hold sums over the 4 haplotypes from each pop in each ind
		for hap1 in xrange(4):								# index of haplotype in first spot
			for hap2 in xrange(4):							# index of haplotype in second spot
				for ind, icode in enumerate(code):   # individuals
					if icode == G[4 * hap1 + hap2]:   # if the current pair of haplotypes is consistent with the given genotype
						for z1 in xrange(n_pops):					 # source pop of hap1
							for z2 in xrange(n_pops):				 # source pop of hap2
								raw = Q[ind, z1] * H[z1, hap1] * Q[ind, z2] * H[z2, hap2]
								isum[ind, z1, hap1] += raw
								isum[ind, z2, hap2] += raw
								norm[ind] += raw
		norm[norm == 0] = 1 # avoid the division by zero due to missing data
		# normalized sum over individuals
		post = np.zeros((n_pops, 4))
		for ind in xrange(n_ind):
			for z in xrange(n_pops):
				for hap in xrange(4):
					#update post for each hap in each pop
					post[z, hap] += isum[ind, z, hap]/norm[ind] #  can we use this estimate an 'effective sample size?'

		# scale the sums so they sum to one  - now represents the haplotype frequencies within pops
		H = np.zeros((n_pops, 4))
		for z in xrange(n_pops):
			H[z] = post[z]/np.sum(post[z])

		#for h in H:
		#	for j in h:
		#		print(j)

		new_LL = get_LL_numba(Q = Q, H = H , code = code)
		delta_LL = new_LL - old_LL

		#if not (delta_LL >= 0):
		#	if (np.abs(delta_LL/old_LL) < 1e-6):
		#		break
		#	else:
		#		assert False, 'delta error'

		if delta_LL <= tol:
			break
		old_LL = new_LL

	return(H, new_LL, i)


#@jit(#"void(int32[:, :], i1[:, :], f8[:, :], f8[:, :], i2, f8, i8, i8[:])",
@optional_numba_decorator
def multiprocess_EM_inner(pairs_inner, shared_genoMatrix, shared_resMatrix, Q, EM_iter, EM_tol, start_idx, seeds):
	npops = Q.shape[1]
	w = start_idx #  used to index the results matrix
	for i in xrange(len(pairs_inner)):
		pair = pairs_inner[i]
		seed = seeds[i]
		# get genotype codes

		codes = shared_genoMatrix[pair[0]] + 3*shared_genoMatrix[pair[1]]

		H = get_rand_hap_freqs(n = npops, seed = seed)

		# do the EM
		res_EM = do_multiEM((H, Q, codes, EM_iter, EM_tol))

		# fill results matrix
		shared_resMatrix[w,0] = pair[0] # index of first locus
		shared_resMatrix[w,1] = pair[1] # index of second locus
		shared_resMatrix[w,2] = np.sum(codes<9) # count non_missing
		shared_resMatrix[w,3] = res_EM[1] # loglike
		shared_resMatrix[w,4] = res_EM[2] # n_iter
		ix = 0

		#print res_EM[0]
		# fill out the haplotype frequencies
		for pop in xrange(npops):
			for hap in xrange(4):
				shared_resMatrix[w,5+ix] = res_EM[0][pop, hap]
				ix+=1
		w +=1


def multiprocess_EM_outer(pairs_outer, shared_genoMatrix, Q, cpus, EM_iter, EM_tol, seeds):
	# TODO pass seeds along in a better way

	# spread the pairs across cpus
	len_pairs = len(pairs_outer)
	per_thread = int(np.ceil(len_pairs/float(cpus)))
	ix_starts = itertools.chain([i * per_thread for i in xrange(cpus)])

	# split across processes
	jobs = np.split(pairs_outer, np.arange(per_thread, len_pairs, per_thread))

	# make a shared results array
	npops = Q.shape[1]
	res_dim2 = 5 + 4*npops # loc1, loc2, count_non_missing, logL, iters, [hap freqs]
	res = np.zeros((len(pairs_outer), res_dim2), dtype = 'f8')
	sharedArray = multiprocessing.Array(ctypes.c_double, res.flatten(), lock = None)
	shared_resMatrix = np.frombuffer(sharedArray.get_obj(), dtype='f8').reshape(res.shape)
	del res

	# set up processes
	processes = [multiprocessing.Process(target=multiprocess_EM_inner, args=(job, shared_genoMatrix, shared_resMatrix,
			Q, EM_iter, EM_tol, next(ix_starts), seeds)) for job in jobs]

	# Run processes
	for proc in processes:
	    proc.start()
	for proc in processes:
	    proc.join()

	return(shared_resMatrix)


@optional_numba_decorator
def get_sumstats_from_haplotype_freqs(H):
	"""given a set of four haplotype frequencies x populations, returns r^2, D, Dprime, and allele frequencies in each pop"""
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
	Dprime = D/Dmax
	r2 = (D**2)/(pA*pB*pa*pb)
	return(r2, D, Dprime, pA, pB)
