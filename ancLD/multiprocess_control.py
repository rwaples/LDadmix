import multiprocessing
import numpy as np
import itertools
import ctypes
from numba import jit

import EM
import utils

NUMBA_DISABLE_JIT = 0
try:
	from numba import jit
except:
	NUMBA_DISABLE_JIT = 1

@jit(nopython=True, nogil=True, cache=False)
def multiprocess_EM_inner(pairs_inner, shared_genoMatrix, shared_resMatrix, Q, EM_iter,
	EM_tol, start_idx, seeds, EM_accel, EM_stop_haps):
	npops = Q.shape[1]
	w = start_idx #  used to index rows in the results matrix

	for i in range(len(pairs_inner)):
		pair = pairs_inner[i]
		seed = seeds[i]
		# get genotype codes
		codes = shared_genoMatrix[pair[0]] + 3*shared_genoMatrix[pair[1]]
		H = utils.get_rand_hap_freqs(n = npops, seed = seed)

		# do the EM
		if (EM_accel & EM_stop_haps):
			res_EM = EM.do_accelEM_stopfreqs((H, Q, codes, EM_iter, EM_tol))
		elif EM_accel:
			res_EM = EM.do_accelEM((H, Q, codes, EM_iter, EM_tol))
		else:
			res_EM = EM.do_multiEM((H, Q, codes, EM_iter, EM_tol))

		LL = res_EM[1]
		H = res_EM[0]
		#flags = np.zeros(npops)
		flags = utils.flag_maf(H, 0.05)

		# fill results matrix
		shared_resMatrix[w,0] = pair[0] # index of first locus
		shared_resMatrix[w,1] = pair[1] # index of second locus
		shared_resMatrix[w,2] = np.sum(codes<9) # count non_missing
		shared_resMatrix[w,3] = LL # loglike
		shared_resMatrix[w,4] = res_EM[2] # n_iter

		# fill out the flags
		for ix in range(npops):
			shared_resMatrix[w, 5+ix] = flags[ix]

		# fill out the haplotype frequencies
		ix = 0
		for pop in range(npops):
			for hap in range(4):
				shared_resMatrix[w, 5+npops+ix] = H[pop, hap]
				ix+=1
		w +=1


def multiprocess_EM_outer(pairs_outer, shared_genoMatrix, Q, cpus, EM_iter, EM_tol, seeds,
		EM_accel, EM_stop_haps):

	# spread the pairs across cpus
	len_pairs = len(pairs_outer)
	per_thread = int(np.ceil(len_pairs/float(cpus)))
	ix_starts = itertools.chain([i * per_thread for i in range(cpus)])

	# split across processes
	jobs = np.split(pairs_outer, np.arange(per_thread, len_pairs, per_thread))

	# make a shared results array
	npops = Q.shape[1]
	res_dim2 = 5 + 5*npops # loc1, loc2, count_non_missing, logL, iters, [flags],[hap freqs]
	res = np.zeros((len(pairs_outer), res_dim2), dtype = 'f8')
	sharedArray = multiprocessing.Array(ctypes.c_double, res.flatten(), lock = None)
	shared_resMatrix = np.frombuffer(sharedArray.get_obj(), dtype='f8').reshape(res.shape)
	del res
	del sharedArray

	# set up processes
	processes = [multiprocessing.Process(target=multiprocess_EM_inner, args=(job, shared_genoMatrix, shared_resMatrix,
			Q, EM_iter, EM_tol, next(ix_starts), seeds, EM_accel, EM_stop_haps)) for job in jobs]

	# Run processes
	for proc in processes:
	    proc.start()
	try:
		for proc in processes:
		    proc.join()
	except KeyboardInterrupt:
		print ("Keyboard interrupt in main")

	return(shared_resMatrix)
