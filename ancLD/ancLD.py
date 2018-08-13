from __future__ import absolute_import, division, print_function
from builtins import (ascii, bytes, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      str, super, zip)# http://python-future.org/imports.html
#base
import argparse
import multiprocessing
import itertools
import collections
import ctypes
import time
import os


try:
	# make sure that numpy doesn't try to use all cores to vectorize matrix operations
	# this shouldn't be the slow part
	import mkl
	mkl.set_num_threads(1)
except ImportError:
	pass


# external
import numpy as np
import pandas as pd

#internal
import ancLD_funcs as LDadmix


## Profiling
PROFILE = True
if PROFILE:
	import data_profiler as profiler
	import pstats
	p = profiler.Profile(signatures=False)
	p.enable()




## TODO
# deal with a haplotype that is fixed in one population - can complicate the post-processing
# check boundary conditions at the end of iterations - still with likelihood
# are the likes better when the edge cases are pushed to the edge
# maybe add a flag column in output
# find cases where the result depend on the starting conditions - maybe haplotype switching
# maybe a post-processing of the Haplotype freqs to set small values to 0?
# could try rounding and norming
# flag for edge cases - should we allow changing?
# post edge-case likelihoods - how to check, one at a time, or all at once?
# allow output format like STRUCTURE p files
# incorporate simulated data
# common output format for simulated data and analyzed data
# can I support both Python 2 and 3?
#   -  working, except we have we likely have to deal with float precision in pandas_to_csv()
# firm up an example data set
# compare the LD calculations external libraries that work on vcf
# add acknowledgements and a link to the greenland paper - Garrett, Filipe
# optional log file with each run?
# speed up loglike calculation?
# add
# option to ditch the likelihood calculations and use delta in haplotype freqs to stop EM
# parse all relevant command line parameters at the start
# check global variables within numba functions
# include bp/cM positions of each locus
# maybe have a simplified output format
# double check the alleles the freqs pertain to ("The numbers 0-2 represent the number of A2 alleles as specified in the .bim file")
# clean up formatting of the output for single-locus EM
# cmd line flag to disable numba
# optional numba decorator for the single-locus frequency estiamtes


# DEBUG = False
#


# argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Input
parser.add_argument('-Q', type=str, default = None,
	help='path to [Q] file')
parser.add_argument('-G', type=str, default = './data/example_1',
	help='path to plink fileset - looks for *.bed/bim/fam')

# Analysis
parser.add_argument('-L', type=int, default=100000,
	help='maximum number of locus pairs to analyze - set to zero for no limit')
parser.add_argument('-D', type=float, default=np.float(0),
	help='only analyze pairs of sites within [D] distance, set to zero for no limit')
parser.add_argument('-C', action='store_true',
	help='set this flag to use genetic postion of loci - default is bp')
parser.add_argument('-N', action='store_true',
	help='set this flag to use SNP order as the distance measure - default is bp')

# Threading
parser.add_argument('-P', type=int, default=4,
	help='use [P] cpus')

# EM
parser.add_argument('-S', type=int, default=0,
	help='random number seed, used to initialize haplotype frequencies')
parser.add_argument('-I', type=int, default=100,
	help='max number of EM iterations')
parser.add_argument('-T', type=float, default=1e-6,
	help='EM stopping tolerance, in loglike units')

parser.add_argument('-J', action='store_true',
	help='set this flag to disable numba JIT compilation')
	#NUMBA_DISABLE_JIT = 0


# Output
parser.add_argument('-O', type=str, default = '../scratch/example_1.out',
	help='path to output file')
#parser.add_argument('-R', type=int, default=3, help='Output precision')
parser.add_argument('-B', type=int, default=1000000,
	help='Batch size, the number of pairs to analyze between each write to disk.')
parser.add_argument('-Z', action='store_true',
	help='set flag to gzip output file')
parser.add_argument('-F', action='store_true',
	help='set this flag to estimate allele freqeuncies within each amixture component')

args = parser.parse_args()

#if PROFILE:
#	args.P = 10
#	args.G = './testdata/OUTofAFRICA50_Ryan'
#	args.L = 500000
#	args.D = 500000
#	args.O = './testdata/OUTofAFRICA50_Ryan.TOOCLOSE.ld'

print("\n------------------\nParameters: ")
print("Admixture (Q) file: {}".format(args.Q))
print("Plink fileset: {}".format(args.G))
print("Output file: {}".format(args.O))
print("gzip output? : {}".format(args.Z))

print("Max allowed number of locus pairs: {}  (0 = no limit)".format(args.L))
print("Max allowed distance between locus pairs: {}  (0 = no limit)".format(args.D))

if args.C and args.N:
	assert(False), "Please pick one distance measure, you set both the -C and -N flags"
if args.C:
	print("Distance unit: genetic")
if args.N:
	print("Distance unit: # of SNPs apart")
else:
	print("Distance unit: bp")

print("Number of threads: {}".format(args.P))
SEED = args.S
if SEED == 0:
	SEED = np.random.randint(1, 2**32 - 1) # 32bit int
print("Random seed: {}".format(SEED))

print("Max number of EM iterations: {}".format(args.I))
print("EM tolerance: {}".format(args.T))

# allow disable of numba
if args.J:
	NUMBA_DISABLE_JIT = 1
	print("numba JIT compilation disabled, analysis will be (much) slower".format(args.T))

print("------------------\n")

DISTANCE_THRESHOLD = float(args.D) # allowable distance between loci
PAIR_LIMIT = int(args.L)
EM_ITER_LIMIT = args.I
EM_TOL = args.T
BATCH_SIZE = args.B
THREADS = args.P
GZIP = args.Z

if GZIP:
	OUTPATH = args.O + ".gz"
else:
	OUTPATH = args.O

print("\n------------------\nLoading data:")
sample_list, locus_list, geno_array, HAS_MISSING = LDadmix.load_plinkfile(args.G)
# bookkeeping
data_nloci, data_nsamples = geno_array.shape
print("Shape of genotype data:\n\t{}\tloci\n\t{}\tindividuals".format(data_nloci, data_nsamples))


if args.Q is None:
	print("No Q matrix was specified, assuming a single population.")
	# Make a Q matrix with just one pop
	q = np.ones((data_nsamples, 1))
else:
	q = pd.read_csv(args.Q, header = None, sep = None, engine = 'python') # hopefully this should allow space and tab-delimited files
	q = q.values
print("Shape of Q data:\n\t{}\tindividuals\n\t{}\tpopulations".format(q.shape[0], q.shape[1]))
print("\tMean of the admixture components: {}".format(q.mean(axis = 0)))

# quick sanity check
assert q.shape[0] == data_nsamples, (
	"The number of individuals in the Q file doesn't match the G file!")
# bookkeeping
NPOPS = q.shape[1]

print("\nDone loading data")
print("------------------\n")


print("\n------------------")
print("Setting up the analysis")

# check for multiple chromosomes here
seen_chromosomes = set([xx.chromosome for xx in locus_list])
print("\nFound loci on {} different chromosome(s)".format(len(seen_chromosomes),
	','.join([str(xx) for xx in seen_chromosomes])))


nloci_on_chr = dict()  # number of loci on each chromosome
loci_on_chr = dict()   # loci on each chromosome - these are
for CHR in seen_chromosomes:
	loci_on_chr[CHR] = list(filter(lambda locus: locus.chromosome == CHR, locus_list))
	nloci_on_chr[CHR] = len(loci_on_chr[CHR])


for (CHR, count) in nloci_on_chr.items():
	print ("\t{:>11,} loci on chromosome {}".format(count, CHR))

print ("\n")


#  use an array of sequential seeds
shared_seeds = multiprocessing.Array(ctypes.c_int32, np.arange(SEED, SEED+BATCH_SIZE), lock = None)
shared_seeds_np = np.frombuffer(shared_seeds.get_obj(), dtype='int32')

# make a shared array for the Q values
array_dim = q.shape
shared_q_array = multiprocessing.Array(ctypes.c_double, q.flatten(), lock = None)
shared_q_matrix = np.frombuffer(shared_q_array.get_obj(), dtype='f8').reshape(array_dim)

# main LD analysis
if not args.F:
	SEEN_PAIRS = 0
	FIRST = True
	# main analysis loop over chromosomes
	for CHR in seen_chromosomes:
		# get the locus pairs we need to analyze
		print ("Start CHR: {}".format(CHR))
		possible_pairs = int((nloci_on_chr[CHR]*(nloci_on_chr[CHR]-1))/2)

		# positions of loci on the chromosome
		positions = np.array([locus.bp_position for locus in loci_on_chr[CHR]]) # default to bp position

		#print(positions)

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
			t1 = time.time()
			batch_EM_res = LDadmix.multiprocess_EM_outer(pairs_outer=batch, shared_genoMatrix=shared_geno_matrix, Q=shared_q_matrix, cpus=THREADS,
				EM_iter = EM_ITER_LIMIT, EM_tol = EM_TOL, seeds = shared_seeds_np)
			t2 = time.time()
			print("\t\tfinished in {:.6} \tseconds, writing to disk".format(t2-t1))

			# make output
			pop_dfs = []
			for popix in range(NPOPS):
				pop_df = pd.concat([
					pd.DataFrame(batch_EM_res[:, 0:5]), # metadata
					pd.DataFrame(batch_EM_res[:, 5+popix]), # flag
					pd.DataFrame(batch_EM_res[:, 5+NPOPS+4*popix: 9+NPOPS+popix*4]) # haplotype freqs
					], axis = 1)

				pop_df.columns = ['i1', 'i2', 'non_missing', 'logLike', 'iter', 'flag', 'Hap00', 'Hap01', 'Hap10', 'Hap11']
				pop_df[['i1', 'i2', 'non_missing', 'iter', 'flag']] = pop_df[['i1', 'i2', 'non_missing', 'iter', 'flag']].astype(np.int)
				r2, D, Dprime, pA, pB = LDadmix.get_sumstats_from_haplotype_freqs(batch_EM_res[:, 5+NPOPS+4*popix: 9+NPOPS+popix*4]) # LD for each pop
				# distance between loci
				bp_dist =  [np.abs(loci_on_chr[CHR][i].bp_position - loci_on_chr[CHR][j].bp_position) for (i,j) in zip(pop_df['i1'], pop_df['i2'])]
				genetic_dist = [np.abs(loci_on_chr[CHR][i].position - loci_on_chr[CHR][j].position) for (i,j) in zip(pop_df['i1'], pop_df['i2'])]

				pop_df['CHR'] = CHR
				pop_df['locus1'] = [loci_on_chr[CHR][i].name for i in pop_df['i1']] #  is this wrong?
				pop_df['locus2'] = [loci_on_chr[CHR][i].name for i in pop_df['i2']]
				pop_df['bp_dist'] = bp_dist
				pop_df['genetic_dist'] = genetic_dist
				pop_df['pop'] = popix+1 # start pop numbering at 1
				pop_df['r2'] = r2
				pop_df['D'] = D
				pop_df['Dprime'] = Dprime
				pop_df['p1'] = pA
				pop_df['p2'] = pB
				pop_dfs.append(pop_df)

			batch_EM_df = pd.concat(pop_dfs)
			batch_EM_df = batch_EM_df[['i1', 'i2', 'locus1', 'locus2',  'CHR', 'bp_dist', 'genetic_dist', 'non_missing', 'pop', 'iter',
				'logLike', 'flag', 'Hap00', 'Hap01', 'Hap10', 'Hap11', 'r2', 'D', 'Dprime', 'p1', 'p2']].sort_values(['i1', 'i2', 'pop'])
			formats = ['%d', '%d', '%s', '%s', '%s', '%d', '%g', '%d', '%d', '%d',
				'%.9f', '%d', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f']


			if GZIP: # compress the output/
				compression = 'gzip'
			else:
				compression = None
			if FIRST: # clobber and write the header
				#header = True
				mode = 'w'
			else: # append and no header
				#header = None
				mode = 'a'

			#batch_EM_df.to_csv(OUTPATH, index = None, header = header, mode = mode,
			#	sep = '\t', na_rep = 'NA',
			#	chunksize=len(batch_EM_df),
			#	compression = compression, float_format = '%g')

			#formats = []
			#for ii in range(len(batch_EM_df.columns)):
			#	coltype = batch_EM_df[batch_EM_df.columns[ii]].dtype
			#	ff = '%s'
			#	if coltype == np.int64:
			#		ff = '%d'
			#	elif coltype == np.float64:
			#		ff = '%f'
			#	formats.append(ff)
			LDadmix.df2csv(df=batch_EM_df, fname=OUTPATH+'_2', formats = formats, mode=mode)


			FIRST = False
			del batch_EM_df, pop_df, pop_dfs, bp_dist, genetic_dist, r2, D, Dprime, pA, pB # cleanup - does this help?

		if STOP:
			break
		else:
			SEEN_PAIRS += n_analysis_pairs


if args.F:
	print("Doing single locus analysis!")
	import mkl
	mkl.set_num_threads(1) # prevents the single-locus likelihood dot products from running amok.

	array_dim = geno_array.shape
	shared_geno_array = multiprocessing.Array(ctypes.c_int8, geno_array.flatten(), lock = None)
	shared_geno_matrix = np.frombuffer(shared_geno_array.get_obj(), dtype='i1').reshape(array_dim)
	del geno_array

	batches = np.split(range(data_nloci), np.arange(BATCH_SIZE, data_nloci, BATCH_SIZE))
	print("There will be {} output batch(es) with up to {:,} loci each".format(len(batches), BATCH_SIZE))
	for count, batch in enumerate(batches, start = 1): # now start batch numbers at 1
		print("\tStarting batch {}".format(count))

		t1 = time.time()
		batch_EM_res = LDadmix.multiprocess_onelocusEM_outer(genos_outer=batch, shared_genoMatrix=shared_geno_matrix, Q=shared_q_matrix, cpus=THREADS,
			EM_iter = EM_ITER_LIMIT, EM_tol = EM_TOL, seeds = shared_seeds_np, bootstraps = 0)
		print ("size of batch_EM_res:")
		print (batch_EM_res.shape)

		t2 = time.time()
		print ("\t\tfinished in {:.6} seconds, writing to disk").format(t2-t1)
		np.savetxt('/home/ryan/LDadmix/testdata/text_singlelocus2.txt', batch_EM_res, delimiter = '\t', fmt = ['%9i', '%9i','%9i', '%1.8f', '%1.6f', '%1.6f', '%1.6f', '%1.6f', '%1.6f', '%1.6f'])

print("\nAll Done!")
print("------------------")
print("output file name: {}".format(OUTPATH))
output_size_bytes = os.path.getsize(OUTPATH)
output_size_megabytes = output_size_bytes/1e6
print("output file size: {:.4} MB".format(output_size_megabytes))


if PROFILE:
	p.disable()
	sortby = 'cumulative'
	ps = pstats.Stats(p)
	with open(OUTPATH+'.profile', 'w') as OUTFILE:
		ps.stream = OUTFILE
		ps.strip_dirs().sort_stats('cumulative').print_stats()
