@jit(nopython=True, nogil=False, cache=True)
def find_boundaries(hrow, zero_threshold ):
	"""give a set of four haplotype frequencies (a row of H),
	return the possible egde case or None"""
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

@jit(nopython=True, nogil=False, cache=True)
def check_boundaries(H, Q, LL, codes, zero_threshold):
	# boundaries within one pop
	# Hap00	Hap01	Hap10	Hap11
	# [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0] # fixed haplotype # both alleles fixed
	# [0, 0, x, 1-x], [0, x, 0, 1-x], [ x, 1-x, 0, 0], [x, 0, 1-x, 0] # fixed allele

	FLAG = np.zeros(len(H))
	bestH = H.copy()
	bestLL = LL

	for i in range(len(H)):
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
