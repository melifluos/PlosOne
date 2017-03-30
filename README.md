# PlosOne
Code to generate PlosOne results

Prerequisites

requires the mmh3 package

Uses the minhash data available at 

https://www.dropbox.com/s/sce6qcmbkpjpeuh/plos_one_data.csv?dl=0

pip install mmh3

Assuming you are in the directory of the source code and have cloned the rep.

To build the LSH table

python LSH.py inpath outpath

To generate metrics for the ground truth communities



