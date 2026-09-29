"""Readers that look at a data file as BYTES before anything interprets it.

Everything else in Kosmos meets a dataset through pandas, which decides on
sight that line 1 is the header and every line under it is data. That decision
is right for most files and silently wrong for the ones this package exists
for: a metabolomics export whose second row is a sample-condition row, a GEO
table with a banner above the header, a matrix with no header at all. When it
is wrong, nothing downstream can tell -- the columns simply have the wrong
names and every column becomes an object dtype -- so the correction has to be
made here, at the point of first contact.
"""
