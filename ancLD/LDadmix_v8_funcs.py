import multiprocessing
import itertools
import numpy as np

NUMBA = False
try:
	from numba import jit
	NUMBA = True
	print("Numba import successful")
except ImportError:
	print("Import of numba failed, code will run slower")

# decorator depending on numba, if numba is found, will decorate with: jit(nopython=True)
def optional_numba_decorator(func):
	if NUMBA:
		return(jit(nopython=True)(func))
	else:
		return(func)


@optional_numba_decorator
def get_rand_hap_freqs(n=2, SEED = None):
	"""returns an (n,4) dimensioned numpy array of random haplotype frequencies,
	where n is the number of pops"""
	if SEED:
		np.random.seed(SEED)

	else:
		pass
	res = np.zeros((n, 4))
	for i in xrange(n):
		res[i] = np.diff(np.concatenate((np.array([0]), np.sort(np.random.rand(3)), np.array([1.0]))))
	return(res)


@optional_numba_decorator
def get_geno_codes(genos):
	"""turns each pair of genotypes into an integer from 0 to 8"""
	return(genos[0] + 3*genos[1])


@optional_numba_decorator
def get_LL_numba(Q, H, code):
	"""returns the loglikelihood of the genotype data given Q and H
	Q = admixture fractions
	H = estimates of source-specific haplotype frequencies"""

	ind_hap_freqs = np.dot(Q, H)
	LL = 0.0
	# had to index the tuples returned by np.where
	LL += np.log(    ind_hap_freqs[np.where(code == 0)[0], 0] * ind_hap_freqs[np.where(code == 0)[0], 0]).sum()
	LL += np.log(2 * ind_hap_freqs[np.where(code == 1)[0], 0] * ind_hap_freqs[np.where(code == 1)[0], 2]).sum()
	LL += np.log(    ind_hap_freqs[np.where(code == 2)[0], 2] * ind_hap_freqs[np.where(code == 2)[0], 2]).sum()
	LL += np.log(2 * ind_hap_freqs[np.where(code == 3)[0], 0] * ind_hap_freqs[np.where(code == 3)[0], 1]).sum()
	LL += np.log(2 * ind_hap_freqs[np.where(code == 4)[0], 0] * ind_hap_freqs[np.where(code == 4)[0], 3]
	           + 2 * ind_hap_freqs[np.where(code == 4)[0], 1] * ind_hap_freqs[np.where(code == 4)[0], 2]).sum()
	LL += np.log(2 * ind_hap_freqs[np.where(code == 5)[0], 2] * ind_hap_freqs[np.where(code == 5)[0], 3]).sum()
	LL += np.log(    ind_hap_freqs[np.where(code == 6)[0], 1] * ind_hap_freqs[np.where(code == 6)[0], 1]).sum()
	LL += np.log(2 * ind_hap_freqs[np.where(code == 7)[0], 1] * ind_hap_freqs[np.where(code == 7)[0], 3]).sum()
	LL += np.log(    ind_hap_freqs[np.where(code == 8)[0], 3] * ind_hap_freqs[np.where(code == 8)[0], 3]).sum()
	return(LL)


@optional_numba_decorator
def do_multiEM(inputs):
    """"""
    H, Q, code, max_iter, tol = inputs # unpack the input

    n_ind = len(Q)
    n_pops = Q.shape[1]
    G = np.array([0,3,1,4, 3,6,4,7, 1,4,2,5, 4,7,5,8]) # which combinations of haplotypes produce which genotypes

    old_LL = get_LL_numba(Q = Q, H = H , code = code)

    # start iteration here
    for i in xrange(max_iter):
        norm = np.zeros(n_ind)
        isum = np.zeros((n_ind, n_pops, 4)) # hold sums over the 4 haplotypes from each pop in each ind
        for hap1 in xrange(4):                                # index of haplotype in first spot
            for hap2 in xrange(4):                            # index of haplotype in second spot
                for ind, icode in enumerate(code):   # individuals
                    if icode == G[4 * hap1 + hap2]:   # if the current pair of haplotypes is consistent with the given genotype
                        for z1 in xrange(n_pops):                     # source pop of hap1
                            for z2 in xrange(n_pops):                 # source pop of hap2
                                raw = Q[ind, z1] * H[z1, hap1] * Q[ind, z2] * H[z2, hap2]
                                isum[ind, z1, hap1] += raw
                                isum[ind, z2, hap2] += raw
                                norm[ind] += raw

        # normalized sum over individuals
        post = np.zeros((n_pops, 4))
        for ind in xrange(n_ind):
            for z in xrange(n_pops):
                for hap in xrange(4):
                    post[z, hap] += isum[ind, z, hap]/norm[ind] #  can we use this estimate an 'effective sample size?'

        # below doesn't currently work with numba, making the post loop above necessary
        #post = 2*(isum / isum.sum((1,2))[:, np.newaxis,np.newaxis]).sum(0)

        # scale the sums so they sum to one  - now represents the haplotype frequencies within pops
        H = np.zeros((n_pops, 4))
        for p in xrange(n_pops):
            H[p] = post[p]/post[p].sum()

        # again numba doesn't like
        #H = post/(post.sum(1)[:,np.newaxis])

        new_LL = get_LL_numba(Q = Q, H = H , code = code)
        delta_LL = new_LL - old_LL
        assert(delta_LL >= 0)

        if delta_LL < tol:
            break
        old_LL = new_LL

    return(H, new_LL, i)


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
