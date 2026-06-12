# !/bin/bash

# Check if the directory data/html/ exists and has files in it
if [ -d "data/html/" ] && [ "$(ls -A data/html/)" ]; then
    echo "Directory data/html/ exists and has files in it. Skipping scraping."
else
    echo "Directory data/html/ does not exist or is empty. Running scraping script."
    python tack_dataset/tpddb_scraping.py -v
fi

python tack_dataset/tpddb_parsing.py -v
python tack_dataset/curate_protacdb.py -v
python tack_dataset/curate_protacpedia.py -v
python tack_dataset/curate_tpddb.py
python tack_dataset/curate_protacdb_tpddb_protacpedia.py -v
python tack_dataset/data_splitting.py -v
