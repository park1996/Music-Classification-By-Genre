# Music Classification by Genre


## Requirements:
* Python 3.6+
* LibROSA
* NumPy
* Matplotlib
* FFmpeg
* tqdm

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
### Brief:
* The dataset was taken from the Free Music Archive (FMA): https://github.com/mdeff/fma
* The dataset consists of 8000 songs and excellent metadata that includes pre-computed features

### Details:
* The FMA dataset consists of 8 genres: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock
* The metadata contains various information about the songs such as artist, genre, and record date
* Pre-computed features are also part of the metadata such as MFCC, spectral contrast, and Tonnetz
* The training, validation, and test dataset sizes are 6400, 800, and 800 respectively 
* The metadata for all tracks can be downloaded here: [fma_metadata.zip](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip) - size is 342 MB
* A collection of Mel Spectrogram images for the dataset can be found here: [fma_spectrogram.zip](https://drive.google.com/open?id=1mzDKmLba9CooaCL-46H1fmxBD2m4ovhP) - size is 2 GB
* The dataset is located here: [fma_small.zip](https://os.unil.cloud.switch.ch/fma/fma_small.zip) - size is 7.2 GB
* Details about the FMA dataset can be found in the official [paper](https://arxiv.org/pdf/1612.01840.pdf)
### APIs:
The access the dataset data, feel free to use the following APIs:
* feature_extractor class:

|Function Name | Description |
|--------------|-------------|
|get_all_song_ids |	Get all song IDs in the dataset |
|get_genre |	Get the genre for a song |
|get_training_dataset_song_ids |	Get all song IDs in the training dataset |
|get_validation_dataset_song_ids |	Get all song IDs in the validation dataset |
|get_test_dataset_song_ids |	Get all song IDs in the test dataset |
|get_feature |	Get feature for a song |
|get_features_as_nparray |	Get features and return as a numpy array |
|get_echonest_features_as_nparray |	Get Echonest features and return as a numpy array|

* audio_preprocessor class:

|Function Name | Description |
|--------------|-------------|
| save_mel_spectrogram |	Save mel-scaled spectrogram image to a file |
| plot_mel_spectrogram |	Plot mel-scaled spectrogram image to a file |
| get_mel_spectrogram |	Get mel-scaled spectrogram as a numpy array |
| get_mel_spectrogram_with_cache | Load previously saved spectrogram if it exists. If not, it will generate spectrogram and save it as a file.|




