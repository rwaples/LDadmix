import argparse
import multiprocessing
import pathlib
import ctypes
import time
import os

# make sure that numpy doesn't try to use all cores to vectorize matrix operations
try:
	import mkl
	mkl.set_num_threads(1)
except ImportError:
	pass


# external
import numpy as np
import pandas as pd

# internal
import utils
import read_write
import multiprocess_control
import ancLD_singlelocus


# TODO
# firm up an example data set
# default to gzipped output
# add acknowledgements and a link to the greenland paper - Garrett, Filipe
# make a post_processing script
# this would produce an LD-decay output and maybe also a plot with a line per population
# fix single locus analysis
# enforce Numba requirement


# argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Required
parser.add_argument('-G', type=str,
	# default=str(pathlib.Path(__file__).parent / "data/example_1"),
	required=True,
	help='path to plink fileset - looks for *.bed/bim/fam')

# Input
parser.add_argument('-Q', type=str, default=None,
	help='path to Q file (admixture proportions)')

# Analysis
parser.add_argument('-L', type=int, default=1000000,
	help='maximum number of locus pairs to analyze - set to zero for no limit')
parser.add_argument('-D', type=float, default=np.float(0),
	help='only analyze pairs of sites within [D] distance, set to zero for no limit')
parser.add_argument('-C', action='store_true',
	help='set this flag to use genetic postion of loci - default is bp')
parser.add_argument('-N', action='store_true',
	help='set this flag to use SNP order as the distance measure - default is bp')

# Threading
parser.add_argument('-P', type=int, default=4, help='use P cpus')

# EM parameters
parser.add_argument('-S', type=int, default=0,
	help='random number seed, used to initialize haplotype frequencies')
parser.add_argument('-I', type=int, default=200,
	help='max number of EM iterations')
parser.add_argument('-T', type=float, default=1e-5,
	help='EM stop criteria')

# need to implement these flags
parser.add_argument('--like', action='store_true',
	help='set this flag to use delta loglikelihood as the EM stop criteria')
parser.add_argument('-X', action='store_true',
	help='set this flag to disable the accelerated EM')

parser.add_argument('-J', action='store_true',
	help='set this flag to disable numba JIT compilation')

# Output
parser.add_argument('-O', type=str,
	default=str(pathlib.Path(__file__).parent.parent / "scratch/default_output.LDadmix.out.gz"),
	help='path to output file, will gzip compress if it ends with ".gz"')
# parser.add_argument('-R', type=int, default=3, help='Output precision')
parser.add_argument('-B', type=int, default=1000000,
	help='Batch size, the number of pairs to analyze between each write to disk.')
parser.add_argument('-F', action='store_true',
	help='set this flag to estimate allele freqeuncies within each amixture component')

args = parser.parse_args()


print("\n------------------\nParameters: ")
print("Admixture (Q) file: {}".format(args.Q))
print("Plink fileset: {}".format(args.G))
print("Output file: {}".format(args.O))

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
	SEED = np.random.randint(1, 2**32 - 1)  # 32bit int
print("Random seed: {}".format(SEED))

print("Max number of EM iterations: {}".format(args.I))
print("EM tolerance: {}".format(args.T))

# allow disable of numba
if args.J:
	NUMBA_DISABLE_JIT = 1  # flag recognized by numba
	print("numba JIT compilation disabled, analysis will be (much) slower".format(args.T))

print("------------------\n")

DISTANCE_THRESHOLD = float(args.D)  # allowable distance between loci
PAIR_LIMIT = int(args.L)
EM_ITER_LIMIT = args.I
EM_TOL = args.T
EM_ACCEL = True
if args.X:
	EM_ACCEL = False
EM_STOP_HAPS = True
if args.like:
	EM_STOP_HAPS = False
BATCH_SIZE = args.B
THREADS = args.P
OUTPATH = args.O

print("\n------------------\nLoading data:")
samples_df, loci_df, geno_array, HAS_MISSING = read_write.read_plink_pandas(args.G)

# bookkeeping
data_nloci, data_nsamples = geno_array.shape
assert data_nloci == len(loci_df), "The number of loci doesn't match!"
assert data_nsamples == len(samples_df), "The number of samples doesn't match!"
print("Shape of genotype data:\n\t{}\tloci\n\t{}\tindividuals".format(data_nloci, data_nsamples))

if args.Q is None:
	print("No Q matrix was specified, assuming a single population.")
	# Make a Q matrix with just one pop
	q = np.ones((data_nsamples, 1))
else:
	# hopefully this should accomidate both space and tab-delimited files
	q = pd.read_csv(args.Q, header=None, sep=None, engine='python')
	q = q.values
print("Shape of Q data:\n\t{}\tindividuals\n\t{}\tpopulations".format(q.shape[0], q.shape[1]))
print("\tMean of the admixture components: {}".format(q.mean(axis=0)))

# quick sanity check
assert q.shape[0] == data_nsamples, (
	"The number of individuals in the Q file doesn't match the G file!")
# bookkeeping
NPOPS = q.shape[1]

print("\nDone loading data")
print("------------------\n")


print("\n------------------")
print("Setting up the analysis")


# see if the chromosomes can be interpreted as ints
try:
	loci_df['chrom'] = loci_df['chrom'].astype(int)
except ValueError:
	pass

# supply a within-chromosome index for each locus
loci_df['chrom_idx'] = loci_df.groupby(['chrom']).cumcount() + 1
# check for multiple chromosomes here
seen_chromosomes = sorted(list(set(loci_df['chrom'])))
# check if all chromosome names can be interpreted as inputs

print("\nFound loci on {} different chromosome(s)".format(len(seen_chromosomes),
	','.join([str(xx) for xx in seen_chromosomes])))

nloci_on_chr = dict()  # number of loci on each chromosome
loci_on_chr = dict()   # loci on each chromosome - these are
for CHR in seen_chromosomes:
	loci_on_chr[CHR] = loci_df.query('chrom == @CHR')
	nloci_on_chr[CHR] = len(loci_on_chr[CHR])

for (CHR, count) in nloci_on_chr.items():
	print("\t{:>11,} loci on chromosome {}".format(count, CHR))
print("\n")


#  use an array of sequential seeds
shared_seeds = multiprocessing.Array(ctypes.c_int32, np.arange(SEED, SEED + BATCH_SIZE), lock=None)
shared_seeds_np = np.frombuffer(shared_seeds.get_obj(), dtype='int32')

# make a shared array for the Q values
array_dim = q.shape
shared_q_array = multiprocessing.Array(ctypes.c_double, q.flatten(), lock=None)
shared_q_matrix = np.frombuffer(shared_q_array.get_obj(), dtype='f8').reshape(array_dim)


# main LD analysis
if not args.F:
	SEEN_PAIRS = 0
	FIRST = True
	# main analysis loop over chromosomes
	for CHR in seen_chromosomes:
		# get the locus pairs we need to analyze
		print("Start CHR: {}".format(CHR))
		possible_pairs = int((nloci_on_chr[CHR] * (nloci_on_chr[CHR] - 1)) / 2)
		chr_loci = loci_df.query('chrom == @CHR').set_index('chrom_idx')
		# positions of loci on the chromosome, defaults to bp
		positions = chr_loci['pos'].values
		if DISTANCE_THRESHOLD:
			if args.C:
				positions = chr_loci['cm'].values  # use cM position
			if args.N:
				positions = chr_loci['i'].values  # use the number of SNPs

		# ensure the positions are monotonically increasing
		assert np.all(np.diff(positions) >= 0)

		# could skip if there is no distance threshold, but it is decently fast
		analysis_pairs = np.fromiter(
			utils.find_idxpairs_maxdist_generator(positions, DISTANCE_THRESHOLD),
			dtype='int32').reshape(-1, 2)
		n_analysis_pairs = len(analysis_pairs)

		# if a pair_limit is set, restrict the analysis to the remaining pairs and set to stop
		STOP = False
		if PAIR_LIMIT and ((n_analysis_pairs + SEEN_PAIRS) > PAIR_LIMIT):
			analysis_pairs = analysis_pairs[:(PAIR_LIMIT - SEEN_PAIRS)]
			n_analysis_pairs = len(analysis_pairs)
			STOP = True

		print("\tWill analyze {:,} / {:,} possible locus pairs".format(
			n_analysis_pairs, possible_pairs))

		# make a shared geno_array (used by all processes)
		chr_geno_array = geno_array[loci_on_chr[CHR]['i'].values]
		array_dim = chr_geno_array.shape
		shared_geno_array = multiprocessing.Array(
			ctypes.c_int8, chr_geno_array.flatten(), lock=None)
		shared_geno_matrix = np.frombuffer(
			shared_geno_array.get_obj(), dtype='i1').reshape(array_dim)
		del chr_geno_array
		del shared_geno_array

		# if analysis_pairs > requested batch size, break across multiple runs
		batches = np.split(analysis_pairs, np.arange(BATCH_SIZE, n_analysis_pairs, BATCH_SIZE))
		print("\tCHR {} will have {} output batch(es) with up to {:,} pairs each".format(
			CHR, len(batches), BATCH_SIZE))
		for count, batch in enumerate(batches, start=1):  # batch numbers start at 1
			print("\tStarting batch {}".format(count))
			# do the EM
			t1 = time.time()
			batch_EM_res = multiprocess_control.multiprocess_EM_outer(
				pairs_outer=batch,
				shared_genoMatrix=shared_geno_matrix,
				Q=shared_q_matrix,
				cpus=THREADS,
				EM_iter=EM_ITER_LIMIT,
				EM_tol=EM_TOL,
				seeds=shared_seeds_np,
				EM_accel=EM_ACCEL,
				EM_stop_haps=EM_STOP_HAPS
			)

			t2 = time.time()
			print("\t\tfinished in {:.6} \tseconds, writing to disk".format(t2 - t1))

			# get the locus-specifc values
			locus1_idx = batch_EM_res[:, 0]
			locus2_idx = batch_EM_res[:, 1]
			locus1_df = chr_loci.iloc[locus1_idx]
			locus2_df = chr_loci.iloc[locus2_idx]
			locus1_name = locus1_df['snp'].values
			locus2_name = locus2_df['snp'].values
			bp_dist = np.abs(locus1_df['pos'].values - locus2_df['pos'].values)
			genetic_dist = np.abs(locus1_df['cm'].values - locus2_df['cm'].values)

			# make output
			pop_dfs = []
			for popix in range(NPOPS):
				pop_df = pd.concat([pd.DataFrame(batch_EM_res[:, 0:5]),  # metadata
					pd.DataFrame(batch_EM_res[:, 5 + popix]),  # flag
					# haplotype freqs
					pd.DataFrame(batch_EM_res[:, 5 + NPOPS + 4 * popix:
						9 + NPOPS + popix * 4])
				], axis=1)

				pop_df.columns = ['i1', 'i2', 'non_missing', 'logLike', 'iter', 'flag',
					'Hap00', 'Hap01', 'Hap10', 'Hap11']
				pop_df[['i1', 'i2', 'non_missing', 'iter',
					'flag']] = pop_df[['i1', 'i2', 'non_missing', 'iter', 'flag']].astype(np.int)
				r2, D, Dprime, pA, pB = utils.get_sumstats_from_haplotype_freqs(
					batch_EM_res[:, 5 + NPOPS + 4 * popix:
						9 + NPOPS + popix * 4])  # LD for each pop
				pop_df['CHR'] = CHR
				pop_df['locus1'] = locus1_name
				pop_df['locus2'] = locus2_name
				pop_df['bp_dist'] = bp_dist
				pop_df['genetic_dist'] = genetic_dist

				pop_df['pop'] = popix + 1  # start pop numbering at 1
				pop_df['r2'] = r2
				pop_df['D'] = D
				pop_df['Dprime'] = Dprime
				pop_df['p1'] = pA
				pop_df['p2'] = pB
				pop_dfs.append(pop_df)

			batch_EM_df = pd.concat(pop_dfs)
			batch_EM_df = batch_EM_df[['i1', 'i2', 'locus1', 'locus2', 'CHR', 'bp_dist',
				'genetic_dist', 'non_missing', 'pop', 'iter', 'logLike', 'flag',
				'Hap00', 'Hap01', 'Hap10', 'Hap11', 'r2', 'D', 'Dprime',
				'p1', 'p2']].sort_values(['i1', 'i2', 'pop'])

			if FIRST:  # clobber file and write header
				mode = 'w'
			else:  # append and no header
				mode = 'a'

			# new, faster function to write the csv file
			read_write.df2csv(df=batch_EM_df, fname=OUTPATH, mode=mode)
			FIRST = False
			# cleanup - does this help?
			del batch_EM_df, pop_df, pop_dfs, bp_dist, genetic_dist, r2, D, Dprime, pA, pB

		if STOP:
			break
		else:
			SEEN_PAIRS += n_analysis_pairs


# Single locus analysis
if args.F:
	print("Doing single locus analysis!")
	n_bootstraps = 0

	FIRST = True

	array_dim = geno_array.shape
	shared_geno_array = multiprocessing.Array(ctypes.c_int8, geno_array.flatten(), lock=None)
	shared_geno_matrix = np.frombuffer(shared_geno_array.get_obj(), dtype='i1').reshape(array_dim)
	del geno_array

	# I could enfore the locus limit here

	batches = np.split(range(data_nloci), np.arange(BATCH_SIZE, data_nloci, BATCH_SIZE))
	print("There will be {} output batch(es) with up to {:,} loci each".format(
		len(batches), BATCH_SIZE))
	for count, batch in enumerate(batches, start=1):  # now start batch numbers at 1
		print("\tStarting batch {}".format(count))

		t1 = time.time()
		batch_EM_res = ancLD_singlelocus.multiprocess_onelocusEM_outer(batch_indexes=batch,
			shared_genoMatrix=shared_geno_matrix, Q=shared_q_matrix, cpus=THREADS,
			EM_iter=EM_ITER_LIMIT, EM_tol=EM_TOL, seeds=shared_seeds_np,
			bootstraps=n_bootstraps)
		print("size of batch_EM_res:")
		print(batch_EM_res.shape)

		t2 = time.time()
		print("\t\tfinished in {:.6} seconds, writing to disk".format(t2 - t1))

		# formats = ['%9i', '%9i', '%9i', '%9i', '%1.8f'] + ['%1.3f', '%1.3f', '%1.3f'] * NPOPS
		batch_EM_df = pd.DataFrame(batch_EM_res)
		popnums = [pop + 1 for pop in range(NPOPS)]
		popheaders = [['pop{}_maf'.format(pop),
			'pop{}_ci_lower'.format(pop), 'pop{}_ci_upper'.format(pop)] for pop in popnums]
		batch_EM_df.columns = ['locus', 'non_missing',
			'allele_count', 'niter', 'LL'] + sum(popheaders, [])
		if FIRST:
			mode = 'w'
		else:
			mode = 'a'
		ancLD_singlelocus.df2csv(df=batch_EM_df, fname=OUTPATH, mode=mode)
		FIRST = False


print("\nAll Done!")
print("------------------")
print("output file name: {}".format(OUTPATH))
output_size_bytes = os.path.getsize(OUTPATH)
output_size_megabytes = output_size_bytes / 1e6
print("output file size: {:.4} MB".format(output_size_megabytes))
