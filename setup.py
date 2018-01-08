from setuptools import setup, find_packages

setup(
    name='ancLD',
    version='0.8.1',
	python_requires='>=2.7',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='Estimate the haplotype frequencies and LD in the ancestral populations of admixed samples.',
    long_description=open('README.md').read(),
    install_requires=[ 'numpy', 'scipy', 'plinkio', 'pandas'],
	extras_require={'numba_jit_compilation': ['numba']},
    url='https://github.com/rwaples/ancLD',
    author='Ryan Waples',
    author_email='ryan.waples@gmail.com'
)
