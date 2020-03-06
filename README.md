# Music Classification by Genre


## Packages:
Python 3.6+, SciPy, LibROSA , Numpy, Matplotlib, seaborn, dotenv, pydot

## Installation:
```
# Install Python 3.6 or greater
$ sudo apt-get install python3.6

# Setup environment
$ ./setup_env.sh
```

## Test:
```
# Run unit tests
$ python3 -m unit_test.py
```

## Dataset:
### Brief
* The dataset for this project was taken from the Free Music Archive (FMA): https://github.com/mdeff/fma
* The dataset consists of thousands of songs and excellent metadata that includes pre-computed features

### Details:
* The datset consists of 8000 songs in 8 genres taken from the FMA dataset
* These genres include Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, and Rock
* The metadata is in csv files which contain various information about the songs such as artist, genre, and record date
* Pre-computed features are part of the metadata such as MFCC, spectral contrast, and Tonnetz
* Details about the FMA dataset can be found in the official [paper](https://arxiv.org/pdf/1612.01840.pdf)
* The metadata for all tracks can be downloaded here: [fma_metadata.zip](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip) - size is 342 MB


