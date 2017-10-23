from setuptools import setup, find_packages

setup(
    name='ancLD',
    version='0.8a',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='Estimates haplotype frequencies in the ancestral populations from genotypes in admixed samples',
    long_description=open('README.md').read(),
    install_requires=[ 'numpy', 'pandas', 'plinkio'],
	extras_require={'numba_jit_compilation': ['numba']},
    url='https://github.com/rwaples/ancLD',
    author='Ryan Waples',
    author_email='ryan.waples@gmail.com'
)
