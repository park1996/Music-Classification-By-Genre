# Music-Classification-Genre


## Packages:
Python 3.6+, SciPy, LibROSA , Numpy, Matplotlib, seaborn, dotenv, pydot

## Installation:
```
# Install Python 3.6 or greater
$ sudo apt-get install python3.6

# Setup environment
$ ./setup_env.sh
```

## Dataset:
### Brief
* The Free Music Archive (FMA) dataset was used for this project: https://github.com/mdeff/fma
* The dataset consists of thousands of free songs and rich metadata used for feature extraction

### Details:
* The datset consists of 30 second clips from 8000 songs in 8 genres
* The metadata is in csv files which contains various information about the songs such as artist, genre, and record date
* Many pre-computed features are part of the meta data like Mel-Frequency Cepstral Coefficients (MFCC) and Chroma
* Details about the FMA dataset can be found in the official [paper](https://arxiv.org/pdf/1612.01840.pdf)
* The dataset can be downloaded here: [fma_small.zip](https://os.unil.cloud.switch.ch/fma/fma_small.zip) - size is 7.2 GB
* The meta-data for all tracks can be downloaded here: [fma_metadata.zip](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip) - size is 342 MB


