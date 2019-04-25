# LDadmix
Estimate the haplotype frequencies and LD in the ancestral populations of admixed samples.

### usage
```
git clone https://github.com/rwaples/ancLD.git
cd ancLD/ancLD
python ancLD.py --help
```

## requirements
works with Python 3.6+

### python packages
 - numpy
 - pandas
 - plinkio (https://github.com/mfranberg/libplinkio) [install with: "pip install plinkio"]
### optional python packages
 - numba (highly recommended, will speed up the analysis ~100 fold)


### output format
Tab-delimited text file, one line per pair of loci per population.
Output will be gzipped (\*.gz) with the -Z flag.  
Output files can get large.

 - i1, i2 : index of locus[1/2] within chromosome (zero-indexed)
 - locus1, locus2 : name of locus[1/2], taken from the .bim file
 - CHR : chromosome
 - bp_dist : basepair distance between the locus1 and locus2
 - genetic_dist : genetic distance between the locus1 and locus2
 - non_missing : the number of individuals with non-missing genotypes
 - pop : population (based on position in Q file) (one-indexed)
 - iter : number of EM iterations
 - logLike : (log)likelihood of the data at the termination of the EM
 - Hap00 : frequency of the 0_0 haplotype
 - Hap01 : frequency of the 0_1 haplotype
 - Hap10 : frequency of the 1_0 haplotype
 - Hap11 : frequency of the 1_1 haplotype
 - r2 : r^2 value, based on the haplotype frequencies
 - D : D value, based on the haplotype frequencies
 - Dprime : D' value, based on the haplotype frequencies = [abs(D/Dmax)]
 - p1 : allele frequency at locus1, based on the haplotype frequencies
 - p2 : allele frequency at locus2, based on the haplotype frequencies
