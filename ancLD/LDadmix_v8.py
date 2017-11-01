#base
import argparse
import sys
import time
import multiprocessing
import itertools

# external
import numpy as np
import scipy.spatial.distance
from plinkio import plinkfile

#internal
import LDadmix_v8_funcs as LDadmix


# ## TODO
#     # deal with missing data - missing genotypes appear as '3' in the genotype matrix
#     # deal with a haplotype that is fixed in one population.
#     # do we need a test for fixed sites?
#     # deal with random seed - difficult due to threading issues
#     # speed up loglike calculation?
#     # incorporate simulated data
#     # common output format for simulated data and analyzed data
#     # Python 3?
#     # firm up an example data set
#     # compare the LD calculations external libraries that work on vcf
#     # add acknowledgements and a link to the greenland paper


# argparse
parser = argparse.ArgumentParser()
# Input
parser.add_argument('-Q', type=str, default = None, help='path to Q file')
parser.add_argument('-G', type=str, default = './data/example_1', help='path to plink bed file')
parser.add_argument('-O', type=str, default = '../scratch/example_1.out',  help='path to output file')
parser.add_argument('-L', type=int, default=100000, help='maximum number of locus pairs to analyze, set to zero for no limit')
parser.add_argument('-D', type=float, default=float(0), help='analyze only pairs of sites within this distance, set to zero for no limit')
parser.add_argument('-C', type=bool, default=False, help='use genetic postion, default is to use bp position')
# Threading
parser.add_argument('-P', type=int, default=4, help='number of threads')
# EM
parser.add_argument('-I', type=int, default=100, help='Max number of EM iterations')
parser.add_argument('-T', type=float, default=1e-3, help='EM stopping tolerance')
# Output
parser.add_argument('-F', type=str, default='LONG', help='Output format')
parser.add_argument('-R', type=int, default=3, help='Output precision')

args = parser.parse_args()



print("\n------------------\nParameters: ")
print("Q file: {}".format(args.Q))
print("Plink files: {}".format(args.G))
print("Output file: {}".format(args.O))
print("Max number of locus pairs: {}  (0 = no limit)".format(args.L))
print("Max distance of locus pairs to analyze: {}".format(args.D))
if args.C:
	print("Distance unit: genetic")
else:
	print("Distance unit: bp")

print("Number of threads: {}".format(args.P))
print("Max number of EM iterations: {}".format(args.I))
print("------------------\n")


def load_plinkfile(basepath):
    plink_file = plinkfile.open(basepath)
    sample_list = plink_file.get_samples()
    locus_list = plink_file.get_loci()
    my_array = np.zeros((len(plink_file.get_loci( )), len(plink_file.get_samples( ))))
    for i, el in enumerate(plink_file):
        my_array[i] = el
	#if 3 in np.unique(my_array):
	#	has_missing = True
		# not sure what to do with missing data yet
    return(sample_list, locus_list, my_array.astype(np.int))


print("\n------------------\nLoading data:")

sample_list, locus_list, geno_array = load_plinkfile(args.G)
print("Shape of genotype data:\n\t{}\tloci\n\t{}\tindividuals".format(geno_array.shape[0], geno_array.shape[1]))

if args.Q is None:
    print("No Q matrix was specified, assuming a single population.")
    # Make a Q matrix with just one pop
    q = np.ones((geno_array.shape[1], 1))
else:
    q = pd.read_csv(args.Q, header = None, sep = ' ')
    q = q.values
print("Shape of Q data:\n\t{}\tindividuals\n\t{}\tpopulations".format(q.shape[0], q.shape[1]))


# quick sanity checks
assert(q.shape[0] == geno_array.shape[1]), "The number of individuals in the Q file doesn't match the G file!"

print("Done loading data, starting LDadmix.")
print("------------------\n")


# ## Analyze

# bookkeeping
data_nloci = geno_array.shape[0]
data_nsamples = geno_array.shape[1]
max_pairs = int(args.L)
max_dist  = float(args.D)

distance_pairs = None
if max_dist:
	if args.C:
		# use position
		positions = np.array([xx.position for xx in locus_list])[:,None]
	else:
		# use bp position
		positions = np.array([xx.bp_position for xx in locus_list])[:,None]
    # make distance matrix
	distm = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(positions))
    # pairs that meet the criteria, remove double comparisons
	distance_pairs = [x for x in zip(*np.where(distm<=max_dist)) if x[0]<x[1]]
	#print('found {} pairs within {} units'.format(len(distance_pairs), max_dist))

# determine how many pairs will be analyzed
if max_dist:
	if max_pairs:
		npairs = min(len(distance_pairs), max_pairs)
	else:
		npairs = len(distance_pairs)
elif max_pairs:
	npairs = min((data_nloci*(data_nloci-1)/2), max_pairs) # minimum of limit
else:
	npairs = (data_nloci*(data_nloci-1)/2)

npops = q.shape[1]

print("\n------------------")
print("Analysis will proceed for {}/{} possible locus pairs.".format(npairs, (data_nloci*(data_nloci-1)/2)))

start_time = time.time()
# make input iterators
Hs = itertools.imap(LDadmix.get_rand_hap_freqs, itertools.repeat(npops, npairs)) # the repeat length will stop after npairs iterations
Qs = itertools.repeat(q, )
if max_dist:
	codes = itertools.imap(LDadmix.get_geno_codes, [(geno_array[x], geno_array[y]) for (x,y) in distance_pairs])
else:
	codes = itertools.imap(LDadmix.get_geno_codes, itertools.combinations(geno_array, 2)) # use all pairs
iter_iter = itertools.repeat(args.I)
tol_iter = itertools.repeat(args.T)
inputs = itertools.izip(Hs, Qs, codes, iter_iter, tol_iter)

# set up for multiprocessing
cpu = args.P
pool = multiprocessing.Pool(processes = cpu)
print("Using {} cpus".format(cpu))
# do the calculations
pool_outputs = pool.map(func = LDadmix.do_multiEM, iterable=inputs)
pool.close() # no more tasks
pool.join()

print('Done!')
print('*** Running time ***')
print("*** {:.2f} seconds ***".format(time.time() - start_time))
print("------------------\n ")

print("\n------------------ ")
print("Writing results file: {}".format(args.O))

if args.F == 'LONG': # write the long-style output (one line per pop/locus pair)
    with open(args.O, 'w') as OUTFILE:
		# write header
		header = ['Locus1', 'Locus2', 'genetic_dist', 'bp_dist', 'Pop', 'r2', 'D', 'Dprime', 'freq1', 'freq2',
			'Hap00', 'Hap01', 'Hap10', 'Hap11', 'loglike', 'nIter']
		OUTFILE.write('\t'.join(header))
		OUTFILE.write('\n')
		# for each locus pair
		if max_dist:
			pairs = itertools.chain(distance_pairs)
		else:
			pairs = itertools.combinations(xrange(data_nloci), 2)
		for res in pool_outputs:
			pair = next(pairs)
			r2, D, Dprime, pA, pB = LDadmix.get_sumstats_from_haplotype_freqs(res[0])
			for pop in xrange(npops):

				# get the locus names here, quick hack
				locusname_1 = locus_list[pair[0]].name
				locusname_2 = locus_list[pair[1]].name
				OUTFILE.write('{}\t{}\t'.format(locusname_1, locusname_2))

				# get the distances here, quick hack
				pair_distance_genetic = np.abs(locus_list[pair[0]].position - locus_list[pair[1]].position)
				pair_distance_bp =  np.abs(locus_list[pair[0]].bp_position - locus_list[pair[1]].bp_position)
				OUTFILE.write('{}\t{}\t'.format(pair_distance_genetic, pair_distance_bp))

				OUTFILE.write('{}\t'.format(pop))
				OUTFILE.write('{:.{precision}}\t'.format(r2[pop], precision = args.R))
				OUTFILE.write('{:.{precision}}\t'.format(D[pop], precision = args.R))
				OUTFILE.write('{:.{precision}}\t'.format(Dprime[pop], precision = args.R))
				OUTFILE.write('{:.{precision}f}\t'.format(pA[pop], precision = args.R))
				OUTFILE.write('{:.{precision}f}\t'.format(pB[pop], precision = args.R))
				for xx in res[0][pop]:
				    OUTFILE.write('{:.{precision}}\t'.format(xx, precision = args.R))
				OUTFILE.write('{:.{precision}}\t'.format(res[1], precision = 10))
				OUTFILE.write('{}'.format(res[2]))
				OUTFILE.write('\n')



print( "Done writing results file, exiting")
print("------------------\n ")
