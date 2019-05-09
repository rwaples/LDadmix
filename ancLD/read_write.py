import pandas_plink
import numpy as np
import pandas as pd

def load_plinkfile(basepath):
	"to be replaced"
	plink_file = plinkfile.open(basepath)
	sample_list = plink_file.get_samples()
	locus_list = plink_file.get_loci()
	#my_array = np.zeros((len(locus_list), len(sample_list)))
	#for i, el in enumerate(plink_file):
	#	my_array[i] = el
	my_array = np.array([sa for sa in plink_file], dtype='i1')

	# look for missing data
	has_missing = False
	if 3 in np.unique(my_array):
		has_missing = True
		# replace missing values with 9
		# this will cause them to never match during genotype code checking
		my_array[my_array == 3] = 9

	my_array = my_array.astype('i1')
	plink_file.close()
	return(sample_list, locus_list, my_array, has_missing)

def read_plink_pandas(basepath):
    bim, fam, G = pandas_plink.read_plink(basepath, verbose = False)
    # G is a dask array
    Gp = np.array(G.compute()) # turn the Dask array into a numpy array
    Gp[np.isnan(Gp)] = 9 # use 9 for missing values, rather than nan
    Gp = Gp.astype('i1')
    return(fam, bim, Gp, (Gp>8).any())


def df2csv(df, fname, formats, mode):
	"""adapted from https://stackoverflow.com/q/15417574"""
	sep = '\t'
	Nd = len(df.columns)
	Nd_1 = Nd - 1
	with open(fname, mode) as OUTFILE:
		# only write heading if clobbering file
		if mode == 'w':
			OUTFILE.write(sep.join(df.columns) + '\n')
		for row in df.itertuples(index=False):
			# is this really the best way to build up strings?
			ss = ''
			for ii in range(Nd):
				ss += formats[ii] % row[ii]
				if ii < Nd_1:
					ss += sep
			OUTFILE.write(ss+'\n')


import sys
if sys.version_info >= (3,0):
	from read_write3 import df2csv
