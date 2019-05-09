import pandas as pd
def df2csv(df, fname, formats, mode):
	"""now with string-literals
	doesn't work with python2"""
	sep = '\t'
	if fname.endswith('.gz'):
		import gzip
		opencmd = gzip.open
		mode = mode+'t' # need to specify text mode
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
