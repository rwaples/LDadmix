
# ## Reworked the two-locus simualtion code to be a bit more clear
# ### still need to re-incorporate it into a broad strategy that allows evalaution of results across many simulation runs

## TODO deal with calculating summary stats

import collections

import numpy as np
import pandas as pd
import scipy.stats


#import LDadmix_v8_funcs as LDadmix
import ancLD_funcs as LDadmix

import ancLD_simulate_funcs as ancLD_sim

class Pop(object):
	def __init__(self, r2, p1, p2):
		"""
		A population with parametric allele frequencies and LD.

		r2 = r2 between sites
		p1 = freq at site 1
		p2 = freq at site 2
		"""
		assert(0 < p1 < 1)
		assert(0 < p2 < 1)
		assert(0 <= r2 <= 1)

		self.r2_parametric = r2
		self.p1_parametric = p1
		self.p2_parametric = p2
		self.hap_freqs_parametric = ancLD_sim.get_hap_freqs_fromLD(r2, p1, p2)
		_r2, _D, _Dprime = ancLD_sim.get_LD(p1, p2, self.hap_freqs_parametric[3])
		if _r2: # not np.nan
			assert(np.allclose(self.r2_parametric, _r2))
		self.Dr2_parametric = _D
		self.Dprimer2_parametric = _Dprime
		# doesn't have haplotype counts or arrays
		self.hap_counts = None
		self.hap_array = None

class Haplotype_sample(Pop):
	"""A sample of n haplotypes from a Pop, and their associated frequencies and LD """
	def __init__(self, r2, p1, p2, n):
		super(Haplotype_sample, self).__init__(r2, p1, p2)
		# gets parametric allele frequencies, LD, as well as haplotype frequencies
		self.n = int(n)
		# 2n haplotypes are sampled, now has hap array and hap counts
		self.hap_array = ancLD_sim.sample_haplotypes(2*self.n, *self.hap_freqs_parametric)
		self.hap_counts = ancLD_sim.get_hap_counts(self.hap_array)
		self.hap_freqs_sample = ancLD_sim.get_hap_freqs(self.hap_array)

		# TODO summary stats of the sampled haplotypes
		self.p1_sample = self.hap_freqs_sample[2] + self.hap_freqs_sample[3]
		self.p2_sample = self.hap_freqs_sample[1] + self.hap_freqs_sample[3]
		self.r2_sample, self.D_sample, self.Dprime_sample  = ancLD_sim.get_LD(self.p1_sample, self.p2_sample, self.hap_freqs_sample[3])


class Admixed_sample(object):
	"""admixed samples generated from two Haplotype_samples"""
	def __init__(self, hs1, hs2, Q, c = 0):
		self.hs1 = hs1
		self.hs2 = hs2
		self.Q = Q # chance the haplotype is from hs2 (and not hs1)
		self.n = len(Q) # haploid size
		self._2n = 2*self.n # diploid size
		self.c = c

		# TODO: check the sample sizes are sufficient
		# make sure the orientation of Q is correct
		assert(self._2n <= self.hs1.n)
		assert(self._2n <= self.hs2.n)

		# determine the origin of each individual's haplotypes, prior to recombination : 2 haplotypes based on Q
		self.source_pops = np.random.binomial(n = 1, p = np.repeat(Q, 2))
		# prior to admixture
		#self.admixed_haplotypes = np.where(self.source_pops[:, np.newaxis], self.hs2.pop_haplotypes, self.hs1.pop_haplotypes)
		self.haplotypes_prior_to_admixture = np.where(self.source_pops[:, np.newaxis], self.hs2.hap_array[:self._2n], self.hs1.hap_array[:self._2n])

		# where to deal with recombination
		self.rec_events = np.random.binomial(n = 1, p = np.repeat(c, self._2n))
		self.no_rec_source_pops = self.source_pops[(1-self.rec_events).astype('bool')] # source pops of the non-recombining haplotypes
		which_keep = np.random.binomial(n = 1, p = np.repeat(0.5, self._2n)) # keep left or right
		which_keep = np.vstack([which_keep, 1-which_keep]).T
		self.raw_sources = np.where(self.rec_events[:, np.newaxis], which_keep, [0,0]) # use either admixed_haplotypes or alternate_haplotypes

		# from each pop
		self.repick_pop = np.random.binomial(n = 1, p = np.repeat(Q, 2))
		repick_site1 = np.random.binomial(n = 1, p = np.where(self.repick_pop, self.hs2.p1_sample, self.hs1.p1_sample))
		repick_site2 = np.random.binomial(n = 1, p = np.where(self.repick_pop, self.hs2.p2_sample, self.hs1.p2_sample))
		self.alternate_haplotypes = np.vstack([repick_site1, repick_site2]).T # still need to combine with admixed, not all are changing
		self.haplotypes_post_admixture = np.where(self.raw_sources, self.alternate_haplotypes, self.haplotypes_prior_to_admixture)
		self.hap_array = self.haplotypes_post_admixture
		# get the frequencies and counts from the resulting haplotypes
		self.hap_freqs_sample = ancLD_sim.get_hap_freqs(self.hap_array)
		self.hap_counts = ancLD_sim.get_hap_counts(self.hap_array)
        
# summarize the haplotypes from each K
		self.hap_freqs_of_k =dict()
		self.hap_counts_of_k =dict()
		for k in [0,1]:
			self.hap_freqs_of_k[k] = ancLD_sim.get_hap_freqs(self.hap_array[self.source_pops==k])
			self.hap_counts_of_k[k] = ancLD_sim.get_hap_counts(self.hap_array[self.source_pops==k])
		#geno codes
		self.geno_codes = LDadmix.get_geno_codes((self.hap_array[::2] + np.roll(self.hap_array,-1, axis = 0)[::2]).T)       
		# TODO summary stats of the admixed haplotypes
		self.rec_count = self.rec_events.sum()
		self.no_rec_count = self._2n - self.rec_events.sum()
