import ctypes
import numpy as np
from numba import jit
import itertools
import multiprocessing


def multiprocess_onelocusEM_outer(batch_indexes, shared_genoMatrix, Q, cpus, EM_iter, EM_tol, seeds, bootstraps):
	"""Outer wrapper for single locus EM."""
	# TODO pass seeds along in a better way

	batch_size = len(batch_indexes)
	per_thread = int(np.ceil(batch_size / float(cpus)))
	ix_starts = itertools.chain([i * per_thread for i in range(cpus)])

	# split across processes
	jobs = np.split(batch_indexes, np.arange(per_thread, batch_size, per_thread))

	# make a shared results array
	npops = Q.shape[1]
	res_dim2 = 5 + (3 * npops)
	res = np.zeros((batch_size, res_dim2), dtype='f8')
	sharedArray = multiprocessing.Array(ctypes.c_double, res.flatten(), lock=None)
	shared_resMatrix = np.frombuffer(sharedArray.get_obj(), dtype='f8').reshape(res.shape)
	print("size of res matrix:")
	print(shared_resMatrix.shape)
	del res

	n = Q.shape[0]
	genos_resample = np.zeros(n, dtype='i1')
	Q_resample = np.zeros((n, Q.shape[1]), dtype='f8')
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
def multiprocess_onelocusEM_inner(genos_inner, shared_genoMatrix, shared_resMatrix, Q, EM_iter,
		EM_tol, start_idx, seeds, bootstraps, genos_resample, Q_resample):
	"""Inner wrapper for single locus EM."""

	npops = Q.shape[1]
	w = start_idx #  used to index the results matrix
	for i in range(len(genos_inner)):
		seed = seeds[i]
		locus = genos_inner[i]
		genos = shared_genoMatrix[locus]
		# exclude the missing genotypes
		# also need to update Q to reflect missing data
		Q = Q[genos<3]
		genos = genos[genos<3]
		non_missing = len(genos)
		assert(len(Q) == len(genos))
		# allele count
		AC = np.sum(genos)
		if (AC == 2*len(genos)) or (AC == 0): # not polymorphic
			dont_EM = True
			maf_em = np.array([np.nan, np.nan])
			niter = 0
			LL = np.nan
		else:
			dont_EM = False
			# do the EM

			maf = np.array([0.5, 0.5]) # only support k=2 for now
			maf_em, niter, LL = emFreqAdmix(genos, Q, maf, maxiter = EM_iter, tol = EM_tol)
			if np.isnan(maf_em[0]):
				# try again
				maf = np.array([np.sum(genos)/2.0, np.sum(genos)/2.0])
				maf_em, niter, LL = emFreqAdmix(genos, Q, maf, maxiter = EM_iter, tol = EM_tol)
				if np.isnan(maf_em[0]):
					# try again
					maf = np.random.rand(2)
					maf_em, niter, LL = emFreqAdmix(genos, Q, maf, maxiter = EM_iter, tol = EM_tol)
			n = Q.shape[0]

		# fill results matrix
		shared_resMatrix[w,0] = locus # index of first locus
		shared_resMatrix[w,1] = non_missing # index of first locus
		shared_resMatrix[w,2] = AC
		shared_resMatrix[w,3] = niter # count non_missing
		shared_resMatrix[w,4] = LL # loglike
		# fill out the maf
		for pop in range(npops):
			shared_resMatrix[w, 5+3*pop] = maf_em[pop]

		if (bootstraps > 0) and (dont_EM == False):
			# calculate the 95% intervals
			alpha = .95
			low_ix, high_ix = np.int(((1-alpha)/2)*bootstraps), np.int((alpha+(1-alpha)/2)*bootstraps)

			res_bootstrap = np.zeros((bootstraps, npops))
			for i in range(bootstraps):
				maf = np.array([0.5, 0.5])
				gr, qr = bootstrap_resample(genos, Q,  genos_resample, Q_resample, n)
				maf_em_bs, niter_bs, LL_bs = emFreqAdmix(gr, qr, maf, maxiter = EM_iter, tol = EM_tol)
				for pop in range(npops):
					res_bootstrap[i, pop] = maf_em_bs[pop]
			for pop in range(npops):
				shared_resMatrix[w, 6+3*pop] = np.sort(res_bootstrap[:,pop])[low_ix] # low
				shared_resMatrix[w, 7+3*pop] = np.sort(res_bootstrap[:,pop])[high_ix] # high
		else:
			for pop in range(npops):
				shared_resMatrix[w, 6+3*pop] = np.nan
				shared_resMatrix[w, 7+3*pop] = np.nan # high
		w +=1








@jit(nopython=True)
def emFreqAdmix(genos, Q, maf, maxiter, tol):
	""" returns an EM estimate of MAF within each admixture component
	"""
	old_LL = ll_FreqAdmix(genos, Q, maf)
	for i in range(1, maxiter+1): # iterations
		for k in range(Q.shape[1]):
			# update the mafs within each pop
			aNorm = np.dot(Q, maf) # dosage from each allele in each ancestry
			bNorm = np.dot(Q, 1.0-maf)
			ag = genos * Q[:,k] * maf[k]/aNorm
			bg = (2-genos) * Q[:,k] * (1.0-maf[k])/bNorm
			fnew = np.sum(ag) / np.sum(ag + bg)
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
	# individual allele frequencies
	iaf = np.dot(Q, maf)
	return(np.log(((1-iaf[genos==0])**2)).sum() +
		np.log(2*(1-iaf[genos==1])*iaf[genos==1]).sum() +
		np.log(((iaf[genos==2])**2)).sum())


@jit(nopython=True)
def bootstrap_resample(genos, Q,  genos_resample, Q_resample, n):
	"bootstrap resampling of (genotypes, Q values)"
	assert(len(genos) == len(Q))
	if n == 0:
		n = len(genos)
		# resample indexes
	resample_ix =  np.rint(np.floor(np.random.rand(n)*n))

	for i in range(n):
		genos_resample[i] = genos[int(resample_ix[i])]
		Q_resample[i] = Q[int(resample_ix[i])]
	return(genos_resample, Q_resample)
