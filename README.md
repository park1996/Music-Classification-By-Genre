# Music Classification by Genre


## Requirements:
* Python 3.6+
* LibROSA
* NumPy
* Matplotlib

## Setup:
```
# Setup environment
$ ./setup_env.sh -i -m
```

## Test:
```
# Run unit tests
$ python3 -m unit_test.py
```

## Dataset:
### Brief
* The dataset was taken from the Free Music Archive (FMA): https://github.com/mdeff/fma
* The dataset consists of 8000 songs and excellent metadata that includes pre-computed features

### Details:
* The FMA dataset consists of 8 genres: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock
* The metadata is in csv files which contain various information about the songs such as artist, genre, and record date
* Pre-computed features are part of the metadata such as MFCC, spectral contrast, and Tonnetz
* The training, validation, and test dataset sizes are 6400, 800, and 800 respectively 
* The metadata for all tracks can be downloaded here: [fma_metadata.zip](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip) - size is 342 MB
* Details about the FMA dataset can be found in the official [paper](https://arxiv.org/pdf/1612.01840.pdf)


