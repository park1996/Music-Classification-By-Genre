#!/bin/bash

# This script will install the required python packages and download the dataset metadata.
# Note: Python 3.6 or greater is required.

pip3 install numpy seaborn matplotlib scipy python-dotenv pydot

if [ ! -f fma_metadata.zip ]; then
	wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
fi

unzip -o fma_metadata.zip
