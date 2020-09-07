import pandas_plink
import numpy as np


def read_plink_pandas(basepath):
    """Read a plink file.
    """
    bim, fam, G = pandas_plink.read_plink(basepath, verbose=False)
    # G is a dask array
    Gp = np.array(G.compute())  # turn the Dask array into a numpy array
    Gp[np.isnan(Gp)] = 9  # use 9 for missing values, rather than nan
    Gp = Gp.astype('i1')
    return(fam, bim, Gp, (Gp > 8).any())


def df2csv(df, fname, mode):
    """Write a DataFrame to dict."""
    sep = '\t'
    if fname.endswith('.gz'):
        import gzip
        opencmd = gzip.open
        mode = mode + 't'  # need to specify text mode
    else:
        opencmd = open
    with opencmd(fname, mode) as OUTFILE:
        if mode.startswith('w'):
            OUTFILE.write(sep.join(df.columns) + '\n')
        for row in df.itertuples(index=False):
            fstring = f"{row[0]:d}\t{row[1]:d}\t{row[2]:s}\t{row[3]:s}\t{row[4]:d}\t\
{row[5]:d}\t\
{row[6]:g}\t\
{row[7]:d}\t\
{row[8]:d}\t\
{row[9]:d}\t\
{row[10]:.9f}\t\
{row[11]:d}\t\
{row[12]:.4f}\t\
{row[13]:.4f}\t\
{row[14]:.4f}\t\
{row[15]:.4f}\t\
{row[16]:.4f}\t\
{row[17]:.4f}\t\
{row[18]:.4f}\t\
{row[19]:.4f}\t\
{row[20]:.4f}\n"
            OUTFILE.write(fstring)
