"""Expectation maximization."""

from numba import jit
import numpy as np
from utils import map2domain, get_LL_numba


@jit(nopython=True, nogil=False, cache=True)
def do_multiEM(inputs):
	"""EM allowing multithreading."""
	H, Q, code, max_iter, tol = inputs  # unpack input
	n_ind, n_pops = Q.shape
	H = map2domain(H, minfreq=0.01)
	# which combinations of haplotypes produce which genotypes
	# genotypes with missing values will not be found
	G = np.array([0, 3, 1, 4, 3, 6, 4, 7, 1, 4, 2, 5, 4, 7, 5, 8])

	old_LL = get_LL_numba(Q=Q, H=H, code=code)

	# start iteration here
	for i in range(1, max_iter + 1):
		# mod = i % 3
		norm = np.zeros(n_ind)
		isum = np.zeros((n_ind, n_pops, 4))  # sums over the 4 haps from each pop in each ind
		for hap1 in range(4):				# index of haplotype in first spot
			for hap2 in range(4):			# index of haplotype in second spot
				for ind, icode in enumerate(code):   # individuals
					# if the current pair of haplotypes is consistent with the given genotype
					if icode == G[4 * hap1 + hap2]:
						for z1 in range(n_pops):			# source pop of hap1
							for z2 in range(n_pops):		# source pop of hap2
								raw = Q[ind, z1] * H[z1, hap1] * Q[ind, z2] * H[z2, hap2]
								isum[ind, z1, hap1] += raw
								isum[ind, z2, hap2] += raw
								norm[ind] += raw
		norm[norm == 0] = 1  # avoid the division by zero due to missing data
		# normalized sum over individuals
		post = np.zeros((n_pops, 4))
		for ind in range(n_ind):
			for z in range(n_pops):
				for hap in range(4):
					# update post for each hap in each pop
					post[z, hap] += isum[ind, z, hap] / norm[ind]  # can we use this estimate an 'effective sample size?'

		# scale the sums so they sum to one  - now represents the haplotype frequencies within pops
		H = np.zeros((n_pops, 4))
		for z in range(n_pops):
			H[z] = post[z] / np.sum(post[z])
		H = map2domain(H, 1e-4)

		# check to end
		new_LL = get_LL_numba(Q=Q, H=H, code=code)
		delta_LL = new_LL - old_LL
		if delta_LL <= tol:
			break
		old_LL = new_LL

	return(H, new_LL, i)


@jit(nopython=True, nogil=False, cache=True)
def do_accelEM(inputs):
	"""Accelerated SQUAREM.

	see: algorithm S3 in doi.org/10.1111/j.1467-9469.2007.00585.x
	"""
	H, Q, code, max_iter, tol = inputs  # unpack input
	n_ind, n_pops = Q.shape
	# which combinations of haplotypes produce which genotypes
	# genotypes with missing values will not be found
	G = np.array([0, 3, 1, 4, 3, 6, 4, 7, 1, 4, 2, 5, 4, 7, 5, 8])

	H = map2domain(H, minfreq=0.01)  # ensure the starting hap freqs are >1%
	old_LL = get_LL_numba(Q=Q, H=H, code=code)

	bigstep = -8
	smallstep = -1
	mstep = 1.5
	Hpast = np.zeros((3, n_pops, 4))
	Hpast[0, :] = H

	# start iteration here
	for i in range(1, max_iter + 1):
		mod = i % 3
		norm = np.zeros(n_ind)
		isum = np.zeros((n_ind, n_pops, 4))  # sums over the 4 haps from each pop in each ind
		for hap1 in range(4):				# index of haplotype in first spot
			for hap2 in range(4):			# index of haplotype in second spot
				for ind, icode in enumerate(code):   # individuals
					# if the current pair of haplotypes is consistent with the given genotype
					if icode == G[4 * hap1 + hap2]:
						for z1 in range(n_pops):			# source pop of hap1
							for z2 in range(n_pops):		# source pop of hap2
								raw = Q[ind, z1] * H[z1, hap1] * Q[ind, z2] * H[z2, hap2]
								isum[ind, z1, hap1] += raw
								isum[ind, z2, hap2] += raw
								norm[ind] += raw
		norm[norm == 0] = 1  # avoid the division by zero due to missing data
		# normalized sum over individuals
		post = np.zeros((n_pops, 4))
		for ind in range(n_ind):
			for z in range(n_pops):
				for hap in range(4):
					# update post for each hap in each pop
					# can we use this estimate an 'effective sample size?'
					post[z, hap] += isum[ind, z, hap] / norm[ind]

		# scale the sums so they sum to one  - now represents the haplotype frequencies within pops
		H = np.zeros((n_pops, 4))
		for z in range(n_pops):
			H[z] = post[z] / np.sum(post[z])
		H = map2domain(H, 1e-4)

		Hpast[mod, :] = H
		if mod == 2:
			r = Hpast[1] - Hpast[0]
			v = Hpast[2] - Hpast[1] - r
			alpha = -1 * np.linalg.norm(r) / np.linalg.norm(v)
			if alpha > smallstep:
				alpha = smallstep  # maybe we should stop the acceleration at this point?
				# smallstep = alpha/mstep
			if alpha < bigstep:  # alpha is
				alpha = bigstep
				bigstep = alpha * mstep
			Hjump = Hpast[0] - (2 * alpha * r) + (v * alpha**2)
			H = map2domain(Hjump, 1e-4)
			old_LL = get_LL_numba(Q=Q, H=H, code=code)

			continue  # ensure at least one more EM step

		# check to end
		new_LL = get_LL_numba(Q=Q, H=H, code=code)
		delta_LL = new_LL - old_LL
		if delta_LL <= tol:
			break
		old_LL = new_LL
	return(H, new_LL, i)


@jit(nopython=True, nogil=False, cache=True)
def do_accelEM_stopfreqs(inputs):
	"""EM with stop criteria based on the change on haplotype frequencies."""
	H, Q, code, max_iter, tol = inputs  # unpack the input
	n_ind, n_pops = Q.shape
	# which combinations of haplotypes produce which genotypes
	G = np.array([0, 3, 1, 4, 3, 6, 4, 7, 1, 4, 2, 5, 4, 7, 5, 8])
	# constrain each inital haplotype frequency to a min of 1% within each pop
	H = map2domain(H, minfreq=0.01)
	old_H = H

	bigstep = -8
	smallstep = -1
	mstep = 1.5
	Hpast = np.zeros((3, n_pops, 4))  # store the three values of H
	Hpast[0, :] = H

	norm = np.zeros(n_ind)
	isum = np.zeros((n_ind, n_pops, 4))
	post = np.zeros((n_pops, 4))
	# start iteration here
	for i in range(1, max_iter + 1):
		mod = i % 3
		norm.fill(0)
		isum.fill(0)  # sums over the 4 haps from each pop in each ind
		for hap1 in range(4):				# index of haplotype in first spot
			for hap2 in range(4):			# index of haplotype in second spot
				for ind, icode in enumerate(code):   # individuals
					# if the current pair of haplotypes is consistent with the given genotype
					if icode == G[4 * hap1 + hap2]:
						for z1 in range(n_pops):			# source pop of hap1
							for z2 in range(n_pops):		# source pop of hap2
								raw = Q[ind, z1] * H[z1, hap1] * Q[ind, z2] * H[z2, hap2]
								isum[ind, z1, hap1] += raw
								isum[ind, z2, hap2] += raw
								norm[ind] += raw
		for j in range(len(norm)):
			if norm[j] == 0:
				norm[j] = 1
		# norm[norm == 0] = 1 # avoid the division by zero due to missing data
		# normalized sum over individuals
		post.fill(0)
		for ind in range(n_ind):
			for z in range(n_pops):
				for hap in range(4):
					# update post for each hap in each pop
					# can we use this estimate an 'effective sample size?'
					post[z, hap] += isum[ind, z, hap] / norm[ind]

		# scale the sums so they sum to one
		# now represents the haplotype frequencies within pops
		H = np.zeros((n_pops, 4))
		for z in range(n_pops):
			H[z] = post[z] / np.sum(post[z])
		H = map2domain(H, 1e-4)
		Hpast[mod, :] = H

		if mod == 2:
			r = Hpast[1] - Hpast[0]
			v = Hpast[2] - Hpast[1] - r
			alpha = -1 * np.linalg.norm(r) / np.linalg.norm(v)  # S3
			if alpha > smallstep:
				alpha = smallstep  # maybe we should stop the acceleration at this point?
				# smallstep = alpha/mstep
			if alpha < bigstep:  # alpha is
				alpha = bigstep
				bigstep = alpha * mstep
			Hjump = Hpast[0] - (2 * alpha * r) + (v * alpha * alpha)
			H = map2domain(Hjump, 1e-4)
			continue  # ensure at least one more EM step

		# check to end
		delta_H = np.max(np.abs(H - old_H))
		if delta_H < tol:
			break
		old_H = H

	LL = get_LL_numba(Q=Q, H=H, code=code)
	return(H, LL, i)
