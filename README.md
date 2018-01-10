# ancLD
Estimate the haplotype frequencies and LD in the ancestral populations of admixed samples.

### usage
```
git clone https://github.com/rwaples/ancLD.git
cd ancLD/ancLD
python LDadmix_v8.py --help
python LDadmix_v8.py
```

## requirements
python 2.7+
### python packages
 - numpy
 - scipy
 - pandas
 - plinkio (https://github.com/mfranberg/libplinkio) [install with: "pip install plinkio
"]
### optional python packages
 - numba (highly recommended, will speed up the analysis ~100 fold)
