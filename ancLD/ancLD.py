#base
import argparse
import multiprocessing
import itertools
import collections
import ctypes
import time

# external
import numpy as np
import pandas as pd

#internal
import ancLD_funcs as LDadmix



# ## TODO
#	# deal with a haplotype that is fixed in one population - can complicate the post-processing
# 	# maybe a post-processing of the Haplotype freqs to set small values to 0?
#	# add check for valid chromosome names
#	# pass random seeds along in a smarter way
#	# incorporate simulated data
#	# common output format for simulated data and analyzed data
#	# Python 3?
#	# firm up an example data set
#	# compare the LD calculations external libraries that work on vcf
#	# add acknowledgements and a link to the greenland paper
#	# create a log file with each run?
#	# speed up loglike calculation?
#	# maybe should ditch the likelihood calculations (maybe optionally) and use delta in haplotype freqs as the stopping condition
#   parse all relevant command line parameters at the start
#   Fix boolean argparses
#   check global variables within numba functions
DEBUG = False
PROFILE = False


# argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Input
parser.add_argument('-Q', type=str, default = None, help='path to [Q] file')
parser.add_argument('-G', type=str, default = './data/example_1', help='path to plink fileset - looks for *.bed/bim/fam')
parser.add_argument('-O', type=str, default = '../scratch/example_1.out',  help='path to output file')
parser.add_argument('-L', type=int, default=100000, help='maximum number of locus pairs to analyze - set to zero for no limit')
parser.add_argument('-D', type=float, default=np.float(0), help='only analyze pairs of sites within [D] distance, set to zero for no limit')
#parser.add_argument('-C', type=bool, default=False, help='set flag to use genetic postion of loci - default is bp')
parser.add_argument('-C', action='store_true', help='set flag to use genetic postion of loci - default is bp')

parser.add_argument('-N', action='store_true', help='set flag to use SNP order as the distance measure - default is bp')
# Threading
parser.add_argument('-P', type=int, default=4, help='use [P] cpus')
# EM
parser.add_argument('-S', type=int, default=0, help='random number seed, used to initialize haplotype frequencies')
parser.add_argument('-I', type=int, default=100, help='max number of EM iterations')
parser.add_argument('-T', type=float, default=1e-6, help='EM stopping tolerance, in loglike units')
# Output
#parser.add_argument('-F', type=str, default='LONG', help='Output format')
#parser.add_argument('-R', type=int, default=3, help='Output precision')
parser.add_argument('-B', type=int, default=1000000, help='Batch size, the number of pairs to analyze between each write to disk.')
parser.add_argument('-Z', action='store_true', help='set flag to gzip output file')

args = parser.parse_args()

if PROFILE:
	args.P = 10
	args.G = './testdata/OUTofAFRICA50_Ryan'
	args.L = 500000
	args.D = 500000
	args.O = './testdata/OUTofAFRICA50_Ryan.TOOCLOSE.ld'

print("\n------------------\nParameters: ")
print("Admixture (Q) file: {}".format(args.Q))
print("Plink fileset: {}".format(args.G))
print("Output file: {}".format(args.O))
print("gzip output? : {}".format(args.Z))

print("Max number of locus pairs: {}  (0 = no limit)".format(args.L))
print("Max distance between locus pairs: {}".format(args.D))

if args.C and args.N:
	assert(False), "Please pick one distance measure, you set both the -C and -N flags"
if args.C:
	print("Distance unit: genetic")
if args.N:
	print("Distance unit: # SNPs apart")
else:
	print("Distance unit: bp")

print("Number of threads: {}".format(args.P))
SEED = args.S
if SEED == 0:
	SEED = np.random.randint(4294967295) # 32bit int
print("Random seed: {}".format(SEED))

print("Max number of EM iterations: {}".format(args.I))
print("EM tolerance: {}".format(args.T))

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
assert(q.shape[0] == data_nsamples), "The number of individuals in the Q file doesn't match the G file!"
# bookkeeping
NPOPS = q.shape[1]

print("\nDone loading data")
print("------------------\n")


print("\n------------------")
print("Setting up the analysis")

# check for multiple chromosomes here
seen_chromosomes = set([xx.chromosome for xx in locus_list])
print("\nFound loci on {} different chromosome(s)".format(len(seen_chromosomes), ','.join([str(xx) for xx in seen_chromosomes])))
ALLOW_MULTIPLE_CHROMOSOMES = True
if not ALLOW_MULTIPLE_CHROMOSOMES:
	assert (False), 'please provide data from a single chromosome'

nloci_on_chr = dict()
loci_on_chr = dict()
for CHR in seen_chromosomes:
	loci_on_chr[CHR] = filter(lambda locus: locus.chromosome == CHR, locus_list)
	nloci_on_chr[CHR] = len(loci_on_chr[CHR])
	#possible_pairs_on_chr[CHR] = (nloci_on_chr[CHR]*(nloci_on_chr[CHR]-1))/2
	#geno_array_of_chr[CHR] = geno_array[np.array([(locus.chromosome == CHR) for locus in locus_list])]

for (CHR, count) in nloci_on_chr.items():
	print ("\t{:>11,} loci on chromosome {}".format(count, CHR))

print ("\n")


#  use sequential seeds
sharedSeeds = multiprocessing.Array(ctypes.c_int32, np.arange(SEED, SEED+BATCH_SIZE), lock = None)
sharedSeeds_np = np.frombuffer(sharedSeeds.get_obj(), dtype='int32')

# make a shared array for the Q values
array_dim = q.shape
shared_qArray = multiprocessing.Array(ctypes.c_double, q.flatten(), lock = None)
shared_qMatrix = np.frombuffer(shared_qArray.get_obj(), dtype='f8').reshape(array_dim)


SEEN_PAIRS = 0
FIRST = True
# main analysis loop over chromosomes
for CHR in seen_chromosomes:
	# get the locus pairs we need to analyze
	print ("Start CHR: {}".format(CHR))
	possible_pairs = (nloci_on_chr[CHR]*(nloci_on_chr[CHR]-1))/2

	# positions of loci on the chromosome
	positions = np.array([locus.bp_position for locus in loci_on_chr[CHR]]) # default to bp position
	if DISTANCE_THRESHOLD:
		if args.C:
			positions = np.array([locus.position for locus in loci_on_chr[CHR]]) # use cM position
		if args.N:
			positions = np.arange(len(loci_on_chr[CHR])) # use the number of SNPs

	assert np.all(np.diff(positions) >=0) #ensure the positions are monotonically increasing (sorted)

	# could do this in a more simple manner if there is no distance threshold, but it is decently fast
	analysis_pairs = np.fromiter(LDadmix.find_idxpairs_maxdist_generator(positions, DISTANCE_THRESHOLD) , dtype = 'int32').reshape(-1,2)
	n_analysis_pairs = len(analysis_pairs)

	# if a pair_limit is set, restrict the analysis to the remaining pairs and set to stop
	STOP = False
	if PAIR_LIMIT and ((n_analysis_pairs + SEEN_PAIRS) > PAIR_LIMIT):
		analysis_pairs = analysis_pairs[:(PAIR_LIMIT-SEEN_PAIRS)]
		n_analysis_pairs = len(analysis_pairs)
		STOP = True

	print ("\tWill analyze {:,} / {:,} possible locus pairs".format( n_analysis_pairs, possible_pairs))

	# make a shared geno_array
	chr_geno_array = geno_array[np.array([(locus.chromosome == CHR) for locus in locus_list])]
	array_dim = chr_geno_array.shape
	shared_genoArray = multiprocessing.Array(ctypes.c_int8, chr_geno_array.flatten(), lock = None)
	shared_genoMatrix = np.frombuffer(shared_genoArray.get_obj(), dtype='i1').reshape(array_dim)
	del chr_geno_array

	# if the length of analysis_pairs is more than the requested batch size, break it up across multiple runs
	batches = np.split(analysis_pairs, np.arange(BATCH_SIZE, n_analysis_pairs, BATCH_SIZE))
	print ("\tCHR {} will have {} output batch(es) with up to {:,} pairs each".format(CHR, len(batches), BATCH_SIZE))
	for count, batch in enumerate(batches, start = 1): # now start batch numbers at 1
		print ("\tStarting batch {}".format(count))
		# do the EM
		t1 = time.time()
		batch_EM_res = LDadmix.multiprocess_EM_outer(pairs_outer=batch, shared_genoMatrix=shared_genoMatrix, Q=shared_qMatrix, cpus=THREADS,
			EM_iter = EM_ITER_LIMIT, EM_tol = EM_TOL, seeds = sharedSeeds_np)
		#print (batch_EM_res)
		t2 = time.time()
		print ("\t\tfinished in {:.4}").format(t2-t1)

		# make output
		pop_dfs = []
		for popix in xrange(NPOPS):
			pop_df = pd.concat([pd.DataFrame(batch_EM_res[:, 0:5]), pd.DataFrame(batch_EM_res[:, 5+popix*4:9+popix*4])], axis = 1)

			pop_df.columns = ['i1', 'i2', 'non_missing', 'logLike', 'iter', 'Hap00', 'Hap01', 'Hap10', 'Hap11']
			pop_df[['i1', 'i2', 'non_missing', 'iter']] = pop_df[['i1', 'i2', 'non_missing', 'iter']].astype(np.int)
			r2, D, Dprime, pA, pB = LDadmix.get_sumstats_from_haplotype_freqs(batch_EM_res[:, 5+popix*4:9+popix*4]) # LD for each pop

			# distance between loci
			bp_dist =  [np.abs(loci_on_chr[CHR][i].bp_position - loci_on_chr[CHR][j].bp_position) for (i,j) in itertools.izip(pop_df['i1'], pop_df['i2'])]
			genetic_dist = [np.abs(loci_on_chr[CHR][i].position - loci_on_chr[CHR][j].position) for (i,j) in itertools.izip(pop_df['i1'], pop_df['i2'])]

			pop_df['CHR'] = CHR
			pop_df['locus1'] = [loci_on_chr[CHR][i].name for i in pop_df['i1']] # this is wrong
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
			'logLike', 'Hap00', 'Hap01', 'Hap10', 'Hap11', 'r2', 'D', 'Dprime', 'p1', 'p2']].sort_values(['i1', 'i2', 'pop'])

		print ("\tDone with batch {}, writing to disk".format(count))

		if GZIP: # compress the output/
			compression = 'gzip'
		else:
			compression = None
		if FIRST: # clobber and write the header
			header = True
			mode = 'w'
		else: # append and no header
			header = None
			mode = 'a'

		batch_EM_df.to_csv(OUTPATH, index = None, header = header, mode = mode, sep = '\t', compression = compression)
		FIRST = False
		del batch_EM_df, pop_df, pop_dfs, bp_dist, genetic_dist, r2, D, Dprime, pA, pB # cleanup - does this help?

	if STOP:
		break
	else:
		SEEN_PAIRS += n_analysis_pairs
