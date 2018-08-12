import multiprocessing
import itertools
import ctypes

import numpy as np
from plinkio import plinkfile
#import pandas as pd

#NUMBA_DISABLE_JIT = 0
NUMBA = False
RESOLVE_EDGE_CASES = 1

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
		return(jit(nopython=True, nogil=False, cache=False)(func))
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
def get_rand_hap_freqs(n=2, seed = 0):
	"""returns an (n,4) dimensioned numpy array of random haplotype frequencies,
	where n is the number of pops"""
	if seed == 0:
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

		flags = np.zeros(npops)
		LL = res_EM[1]
		H = res_EM[0]
		if RESOLVE_EDGE_CASES:
			flags, LL, H = check_boundaries(H=res_EM[0], Q=Q, LL=res_EM[1], codes=codes, zero_threshold = 1e-4)

		# fill results matrix
		shared_resMatrix[w,0] = pair[0] # index of first locus
		shared_resMatrix[w,1] = pair[1] # index of second locus
		shared_resMatrix[w,2] = np.sum(codes<9) # count non_missing
		shared_resMatrix[w,3] = LL # loglike
		shared_resMatrix[w,4] = res_EM[2] # n_iter

		# fill out the flags
		for ix in xrange(npops):
			shared_resMatrix[w, 5+ix] = flags[ix]

		# fill out the haplotype frequencies
		ix = 0
		for pop in xrange(npops):
			for hap in xrange(4):
				shared_resMatrix[w, 5+npops+ix] = H[pop, hap]
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
	res_dim2 = 5 + 5*npops # loc1, loc2, count_non_missing, logL, iters, [flags],[hap freqs]
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


def multiprocess_onelocusEM_outer(genos_outer, shared_genoMatrix, Q, cpus, EM_iter, EM_tol, seeds, bootstraps):
	# TODO pass seeds along in a better way

	# spread the pairs across cpus
	len_genos = len(genos_outer)
	per_thread = int(np.ceil(len_genos/float(cpus)))
	ix_starts = itertools.chain([i * per_thread for i in xrange(cpus)])

	# split across processes
	jobs = np.split(genos_outer, np.arange(per_thread, len_genos, per_thread))

	# make a shared results array
	npops = Q.shape[1]
	res_dim2 = 4+3*npops # loc, count_non_missing, logL, iters, [population_freqs]
	res = np.zeros((len(genos_outer), res_dim2), dtype = 'f8')
	sharedArray = multiprocessing.Array(ctypes.c_double, res.flatten(), lock = None)
	shared_resMatrix = np.frombuffer(sharedArray.get_obj(), dtype='f8').reshape(res.shape)
	print "size of res matrix:"
	print shared_resMatrix.shape
	del res

	n = Q.shape[0]
	genos_resample = np.zeros(n, dtype = 'i1')
	Q_resample = np.zeros((n, Q.shape[1]), dtype = 'f8')
	# set up processes
	processes = [multiprocessing.Process(target=multiprocess_onelocusEM_inner, args=(job, shared_genoMatrix, shared_resMatrix,
			Q, EM_iter, EM_tol, next(ix_starts), seeds, bootstraps, genos_resample, Q_resample)) for job in jobs]

	# Run processes
	for proc in processes:
	    proc.start()
	for proc in processes:
	    proc.join()

	return(shared_resMatrix)

@jit(nopython=True)
def multiprocess_onelocusEM_inner(genos_inner, shared_genoMatrix, shared_resMatrix, Q, EM_iter, EM_tol, start_idx, seeds, bootstraps, genos_resample, Q_resample):
	npops = Q.shape[1]
	w = start_idx #  used to index the results matrix
	for i in xrange(len(genos_inner)):
		locus = genos_inner[i]
		seed = seeds[i]
		# get genotype codes
		codes = shared_genoMatrix[locus]
		non_missing = len(codes[codes<=2])
		# do the EM
		maf = np.array([0.5, 0.5])
		maf_em, niter, LL = emFreqAdmix(codes, Q, maf, maxiter = EM_iter, tol = EM_tol)
		n = Q.shape[0]

		# calculate the 95% intervals
		alpha = .95
		low_ix, high_ix = np.int(((1-alpha)/2)*bootstraps), np.int((alpha+(1-alpha)/2)*bootstraps)

		# fill results matrix
		shared_resMatrix[w,0] = locus # index of first locus
		shared_resMatrix[w,1] = non_missing # index of first locus
		shared_resMatrix[w,2] = niter # count non_missing
		shared_resMatrix[w,3] = LL # loglike
		# fill out the maf
		for pop in xrange(npops):
			shared_resMatrix[w, 4+3*pop] = maf_em[pop]

		if bootstraps > 0:
			res_bootstrap = np.zeros((bootstraps, npops))
			for i in xrange(bootstraps):
				maf = np.array([0.5, 0.5])
				gr, qr = bootstrap_resample(codes, Q,  genos_resample, Q_resample, n)
				maf_em_bs, niter_bs, LL_bs = emFreqAdmix(gr, qr, maf, maxiter = EM_iter, tol = EM_tol)
				for pop in xrange(npops):
					res_bootstrap[i, pop] = maf_em_bs[pop]
			for pop in xrange(npops):
				shared_resMatrix[w, 5+3*pop] = np.sort(res_bootstrap[:,pop])[low_ix] # low
				shared_resMatrix[w, 6+3*pop] = np.sort(res_bootstrap[:,pop])[high_ix] # high
		w +=1



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

@optional_numba_decorator
def fix(H, decimals):
	"""rounds to a certain number of decimals and then norms to sum to 1"""
	Hround = np.round(H, decimals = decimals)
	return Hround / Hround.sum(1)[:, None]

@optional_numba_decorator
def find_boundaries(hrow, zero_threshold ):
	"""give a set of four haplotype frequencies (a row of H), return the possible egde case or None"""
	near_zero = hrow < zero_threshold
	count_near_zero = near_zero.sum()
	if count_near_zero == 3:
		return (np.rint(hrow)) # round to nearest integers
	elif count_near_zero == 2:
		p1 = hrow[2] + hrow[3]
		p2 = hrow[1] + hrow[3]
		if (p1 < zero_threshold):
			hrow[0] = hrow[0] / (hrow[0] + hrow[1])
			hrow[1] = hrow[1] / (hrow[0] + hrow[1])
			hrow[2] = 0
			hrow[3] = 0
			return(hrow)
		elif (p1 > (1.0-zero_threshold)):
			hrow[0] = 0
			hrow[1] = 0
			hrow[2] = hrow[2] / (hrow[2] + hrow[3])
			hrow[3] = hrow[3] / (hrow[2] + hrow[3])
			return(hrow)
		elif (p2 < zero_threshold):
			hrow[0] = hrow[0] / (hrow[0] + hrow[2])
			hrow[1] = 0
			hrow[2] = hrow[2] / (hrow[0] + hrow[2])
			hrow[3] = 0
			return(hrow)
		elif (p2 > (1.0- zero_threshold)):
			hrow[0] = 0
			hrow[1] = hrow[1] / (hrow[1] + hrow[3])
			hrow[2] = 0
			hrow[3] = hrow[3] / (hrow[1] + hrow[3])
			return(hrow)
	else:
		return (None)

@optional_numba_decorator
def check_boundaries(H, Q, LL, codes, zero_threshold):
	# boundaries within one pop
	# Hap00	Hap01	Hap10	Hap11
	# [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0] # fixed haplotype # both alleles fixed
	# [0, 0, x, 1-x], [0, x, 0, 1-x], [ x, 1-x, 0, 0], [x, 0, 1-x, 0] # fixed allele

	FLAG = np.zeros(len(H)) # FLAG 0 = good, 1 = fixed hap, 2 = fixed allele
	bestH = H.copy()
	bestLL = LL

	for i in xrange(len(H)):
		htest = find_boundaries(H[i], zero_threshold=zero_threshold)
		if htest is not None:
			H[i] = htest
			FLAG[i] = 1
	test_ll = get_LL_numba(Q = Q, H = H, code = codes)
	if test_ll >= bestLL:
		bestLL = test_ll
		bestH = H
	else:
		FLAG = np.zeros(len(H))
	return(FLAG, bestLL, bestH)


# will need to deal with missing data - either before or after bootstrapping
@jit(nopython=True)
def emFreqAdmix(genos, Q, maf, maxiter = 200, tol = 1e-6):
	""" returns an EM estimate of MAF within each admixture component
	"""
	old_LL = ll_FreqAdmix(genos, Q, maf)
	for i in xrange(1, maxiter+1): # iterations
		for k in xrange(Q.shape[1]):
			aNorm = np.dot(Q, maf) # individual allele frequencies
			bNorm = np.dot(Q, 1.0-maf)
			ag = genos * Q[:,k] * maf[k]/aNorm
			bg = (2-genos) * Q[:,k] * (1.0-maf[k])/bNorm
			fnew = np.sum( ag )/np.sum(ag +bg )
			maf[k] = fnew

		new_LL = ll_FreqAdmix(genos, Q, maf)
		delta_LL = new_LL - old_LL
		if delta_LL <= tol:
			break
		old_LL = new_LL
	return(maf, i , new_LL)

@jit(nopython=True)
def ll_FreqAdmix(genos, Q, maf):
	"""returns the loglikelihood of the genotype data given Q and maf"""
	iaf = np.dot(Q, maf) # individual allele frequencies
	return np.log(((1-iaf[genos==0])**2)).sum() + np.log(2*(1-iaf[genos==1])*iaf[genos==1]).sum() + np.log(((iaf[genos==2])**2)).sum()


@jit(nopython=True)
def bootstrap_resample(genos, Q,  genos_resample, Q_resample, n):
    assert(len(genos) == len(Q))
    if n == 0:
        n = len(genos)
    # resample indexes
    resample_ix =  np.rint(np.floor(np.random.rand(n)*len(genos)))

    for i in xrange(n):
        genos_resample[i] = genos[int(resample_ix[i])]
        Q_resample[i] = Q[int(resample_ix[i])]
    return(genos_resample, Q_resample)