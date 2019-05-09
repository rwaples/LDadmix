from setuptools import setup, find_packages

setup(
    name='LDadmix',
    version='0.2',
	python_requires='>=2.7',
    packages=find_packages(exclude=['tests*']),
    license='GPL 3.0',
    description='Estimate the haplotype frequencies and LD in the ancestral populations of admixed samples.',
    long_description=open('README.md').read(),
    install_requires=[ 'pandas', 'numpy', 'scipy', 'pandas-plink', 'numba],
	#extras_require={'numba_jit_compilation': ['numba']},
    url='https://github.com/rwaples/LDadmix',
    author='Ryan Waples',
    author_email='ryan.waples@gmail.com'
)
