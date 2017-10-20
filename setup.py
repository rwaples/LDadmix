from setuptools import setup, find_packages

setup(
    name='ancLD',
    version='0.8a',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='Estimates haplotype frequencies in the ancestral populations from genotypes in admixed samples',
    long_description=open('README.md').read(),
    install_requires=[ 'numpy', 'numba', 'pandas', 'plinkio'],
    url='https://github.com/rwaples/ancLD',
    author='Ryan Waples',
    author_email='ryan.waples@gmail.com'
)
