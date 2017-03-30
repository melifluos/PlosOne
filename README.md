# PlosOne
Code to generate PlosOne results

Prerequisites

requires the mmh3 package

pip install mmh3

Uses the minhash data available at 

https://www.dropbox.com/s/sce6qcmbkpjpeuh/plos_one_data.csv?dl=0

Assuming you are in the directory of the source code and have cloned the rep.

To build the LSH table

python LSH.py minhash_data_path LSH_path

To generate metrics for the ground truth communities

python assess_community_quality.py minhash_data_path outpath

To run experimentation

python run_experimentation.py minhash_data_path LSH_path outpath


