
# coding: utf-8

# In[1]:


#base
import argparse
import sys
import time
import multiprocessing
import itertools

# external
import numpy as np
#import pandas as pd
from plinkio import plinkfile

#internal
import LDadmix_v8_funcs as LDadmix


# ## TODO
#     # deal with missing data - missing genotypes appear as '3' in the genotype matrix
#     # put a max distance
#     # deal with haplotype fixed in one population.
#     # deal with random seed - difficult due to threading issues
#     # speed up loglike calculation?
#     # common output format for simulated data and analyzed data
#     # Python 3?
#     # incorporate simulated data
#     # firm up an example data set
#     # compare to external libraries that work on vcf

# # Interface

# In[7]:


parser = argparse.ArgumentParser()
parser.add_argument('-Q', type=str, default = None, help='path to Q file')
parser.add_argument('-G', type=str, default = './data/example_1', help='path to plink bed file')
parser.add_argument('-O', type=str, default = '../scratch/example_1.out',  help='path to output file')
parser.add_argument('-P', type=int, default=4, help='number of threads')
parser.add_argument('-L', type=int, default=20, help='analyze the first L loci')
parser.add_argument('-I', type=int, default=100, help='Max number of EM iterations')
parser.add_argument('-T', type=float, default=1e-3, help='EM stopping tolerance')
parser.add_argument('-F', type=str, default='LONG', help='Output format')
parser.add_argument('-R', type=int, default=3, help='Output precision')


# In[ ]:


import __main__ as main
if hasattr(main, '__file__'): # if not interactive
    args = parser.parse_args()
else: # if interactive (e.g. in notebook)
    #prevents trying to parse the sys.argv[1] of the interactive session
    args = parser.parse_args(['-Q' './explore/prototype/example_1.admixed.Q',
                             '-G', './explore/prototype/example_1.ld.forLDadmix',
                             '-P', '10', '-L', '500', '-I', '200'])


# In[8]:


print("\n------------------\nParameters: ")
print("Q file: {}".format(args.Q))
print("Plink files: {}".format(args.G))
print("Output file: {}".format(args.O))
print("Number of threads: {}".format(args.P))
print("Max number of loci: {}".format(args.L))
print("Max number of iterations: {}".format(args.I))
print("------------------\n")



def load_plinkfile(basepath):
    plink_file = plinkfile.open(basepath)
    sample_list = plink_file.get_samples()
    locus_list = plink_file.get_loci()
    my_array = np.zeros((len(plink_file.get_loci( )), len(plink_file.get_samples( ))))
    for i, el in enumerate(plink_file):
        my_array[i] = el
	if 3 in np.unique(my_array):
		has_missing = True
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
assert(geno_array.shape[0] >= args.L), "You asked for more loci ({}) than are present in the G file ({})!".format(args.L, geno_array.shape[0])

print("Done loading data, starting LDadmix.")
print("------------------\n")


# ## Analyze
start_time = time.time()

nloci = args.L
npairs = (nloci*(nloci-1))/2
npops = q.shape[1]
print("\n------------------")
print("There are {} locus pairs to analyze.".format(npairs))




# make input iterators
Hs = itertools.imap(LDadmix.get_rand_hap_freqs, itertools.repeat(npops, npairs))
Qs = itertools.repeat(q)
codes = itertools.imap(LDadmix.get_geno_codes, itertools.combinations(geno_array[:nloci], 2))
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


# In[14]:


print("\n------------------ ")
print("Writing results file: {}".format(args.O))


# In[15]:

if args.F == 'WIDE': # write the wide-style output (one line per locus pair)
	# out of date
    with open(args.O, 'w') as OUTFILE:
        # write header
        LDheader = ['r2_Pop{}'.format(x) for x in xrange(1, 1+ npops)] + ['D_Pop{}'.format(x) for x in xrange(1, 1+ npops)] + ['Dprime_Pop{}'.format(x) for x in xrange(1, 1+ npops)]
        freqheader = ['p1_Pop{}'.format(x) for x in xrange(1, 1+ npops)] + ['p2_Pop{}'.format(x) for x in xrange(1, 1+ npops)]
        hapheader = ['Hap{}_Pop{}'.format(hap, pop) for hap,pop in zip(
            [1,2,3,4]*npops, [x for x in xrange(1, npops+1) for i in xrange(4)])]
        header = ['Locus1', 'Locus2'] + LDheader + freqheader + hapheader +['LL', 'nIter']
        OUTFILE.write('\t'.join(header))
        OUTFILE.write('\n')

        # for each locus pair
        for pair, res in zip(itertools.combinations(xrange(nloci), 2), pool_outputs):
            OUTFILE.write('{}\t{}'.format(pair[0], pair[1]))
            r2, D, Dprime, pA, pB = LDadmix.get_sumstats_from_haplotype_freqs(res[0])
            for xx in r2:
                OUTFILE.write('\t{}'.format(xx))
            for xx in D:
                OUTFILE.write('\t{}'.format(xx))
            for xx in Dprime:
                OUTFILE.write('\t{}'.format(xx))
            for xx in pA:
                OUTFILE.write('\t{}'.format(xx))
            for xx in pB:
                OUTFILE.write('\t{}'.format(xx))
            # haps
            for xx in res[0].flatten():
                OUTFILE.write('\t{}'.format(xx))
            OUTFILE.write('\t{}'.format(res[1]))
            OUTFILE.write('\t{}'.format(res[2]))
            OUTFILE.write('\n')


# ### New long format for output

# In[16]:


if args.F == 'LONG': # write the long-style output (one line per pop/locus pair)
    with open(args.O, 'w') as OUTFILE:
        # write header
        header = ['Locus1', 'Locus2', 'Pop', 'r2', 'D', 'Dprime', 'freq1', 'freq2',
                  'Hap00', 'Hap01', 'Hap10', 'Hap11', 'loglike', 'nIter']
        OUTFILE.write('\t'.join(header))
        OUTFILE.write('\n')
        # for each locus pair
        pairs = itertools.combinations(xrange(nloci), 2)
        for res in pool_outputs:
            pair = next(pairs)
            r2, D, Dprime, pA, pB = LDadmix.get_sumstats_from_haplotype_freqs(res[0])
            for pop in xrange(npops):
                OUTFILE.write('{}\t{}\t'.format(pair[0], pair[1]))
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


# In[17]:


print( "Done writing results file, exiting")
print("------------------\n ")
