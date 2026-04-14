# !/bin/bash

python tack_dataset/curate_protacdb.py -v
python tack_dataset/curate_protacpedia.py -v
python tack_dataset/curate_tpddb.py
python tack_dataset/curate_protacdb_tpddb_protacpedia.py -v
python tack_dataset/data_splitting.py -v
