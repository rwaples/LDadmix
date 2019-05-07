import data_profiler as profiler
import pstats


# below just for profiling
if PROFILE:
	p.disable()
	sortby = 'cumulative'
	ps = pstats.Stats(p)
	with open(OUTPATH+'.profile_main', 'w') as OUTFILE:
		ps.stream = OUTFILE
		ps.strip_dirs().sort_stats('cumulative').print_stats()




def multiprocess_EM_profile(pairs_outer, shared_genoMatrix, Q, cpus, EM_iter, EM_tol, seeds):
	# out of date
	# spread the pairs across cpus
	len_pairs = len(pairs_outer)
	per_thread = int(np.ceil(len_pairs/float(cpus)))
	ix_starts = itertools.chain([i * per_thread for i in range(cpus)])

	# split across processes
	jobs = np.split(pairs_outer, np.arange(per_thread, len_pairs, per_thread))
	assert len(jobs) == 1
	job = np.array(jobs[0], dtype='i8')
	# make a shared results array
	npops = Q.shape[1]
	res_dim2 = 5 + 5*npops # loc1, loc2, count_non_missing, logL, iters, [flags],[hap freqs]
	res = np.zeros((len(pairs_outer), res_dim2), dtype = 'f8')
	sharedArray = multiprocessing.Array(ctypes.c_double, res.flatten(), lock = None)
	shared_resMatrix = np.frombuffer(sharedArray.get_obj(), dtype='f8').reshape(res.shape)
	del res
	multiprocess_EM_inner(job, shared_genoMatrix, shared_resMatrix,
			Q, EM_iter, EM_tol, next(ix_starts), seeds)
	return(shared_resMatrix)





if PROFILE:
	print("---\nStarting to profile")
	p = profiler.Profile(signatures=True)
	p.enable()
	# do a mini EM loop here
	SEEN_PAIRS = 0
	FIRST = True
	# main analysis loop over chromosomes
	for CHR in seen_chromosomes:
		# get the locus pairs we need to analyze
		print ("Start CHR: {}".format(CHR))
		possible_pairs = int((nloci_on_chr[CHR]*(nloci_on_chr[CHR]-1))/2)

		positions = np.array([locus.bp_position for locus in loci_on_chr[CHR]])
		# positions of loci on the chromosome, defaults to bp
		if DISTANCE_THRESHOLD:
			if args.C:
				positions = np.array([locus.position for locus in loci_on_chr[CHR]]) # use cM position
			if args.N:
				positions = np.arange(nloci_on_chr[CHR]) # use the number of SNPs


		# ensure the positions are monotonically increasing (sorted)
		assert np.all(np.diff(positions) >=0)

		# could skip if there is no distance threshold, but it is decently fast
		analysis_pairs = np.fromiter(LDadmix.find_idxpairs_maxdist_generator(positions, DISTANCE_THRESHOLD) , dtype = 'int32').reshape(-1,2)
		n_analysis_pairs = len(analysis_pairs)

		# if a pair_limit is set, restrict the analysis to the remaining pairs and set to stop
		STOP = False
		if PAIR_LIMIT and ((n_analysis_pairs + SEEN_PAIRS) > PAIR_LIMIT):
			analysis_pairs = analysis_pairs[:(PAIR_LIMIT-SEEN_PAIRS)]
			n_analysis_pairs = len(analysis_pairs)
			STOP = True

		print ("\tWill analyze {:,} / {:,} possible locus pairs".format( n_analysis_pairs, possible_pairs))

		# make a shared geno_array (used by all processes)
		chr_geno_array = geno_array[np.array([(locus.chromosome == CHR) for locus in locus_list])]
		array_dim = chr_geno_array.shape
		shared_geno_array = multiprocessing.Array(ctypes.c_int8, chr_geno_array.flatten(), lock = None)
		shared_geno_matrix = np.frombuffer(shared_geno_array.get_obj(), dtype='i1').reshape(array_dim)
		del chr_geno_array

		# if there are more analysis_pairs than the requested batch size, break it up across multiple runs
		batches = np.split(analysis_pairs, np.arange(BATCH_SIZE, n_analysis_pairs, BATCH_SIZE))
		print ("\tCHR {} will have {} output batch(es) with up to {:,} pairs each".format(CHR, len(batches), BATCH_SIZE))
		for count, batch in enumerate(batches, start = 1): # batch numbers start at 1
			print ("\tStarting batch {}".format(count))
			# do the EM
			batch_EM_res = LDadmix.multiprocess_EM_profile(pairs_outer=batch, shared_genoMatrix=shared_geno_matrix, Q=shared_q_matrix, cpus=THREADS,
				EM_iter = EM_ITER_LIMIT, EM_tol = EM_TOL, seeds = shared_seeds_np)

	p.disable()
	sortby = 'cumulative'
	ps = pstats.Stats(p)
	with open(OUTPATH+'.profile_EM', 'w') as OUTFILE:
		ps.stream = OUTFILE
		ps.strip_dirs().sort_stats('cumulative').print_stats()
