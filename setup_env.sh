#!/bin/bash

# This script will install the required python packages and download the dataset and metadata.

print_help() {
	echo "Usage:"
	echo "$ $0  <options>"
	echo ""
	echo "  options:"
	echo "	-d - Download dataset"
	echo "	-m - Download metadata"
	echo "	-h - This help string"
	echo ""
}

download_meta_data=0
download_dataset=0
metadata_file=fma_metadata.zip
dataset_file=fma_small.zip
help=false


while getopts "hdm" options; do
	case "${options}" in
		d)
			download_dataset=1
			echo "Downloading dataset"
			;;
		m)
			download_metadata=1
			echo "Downloading metadata"
			;;

		h | *)
			print_help
			exit 0
			;;
	esac
done



if [[ $download_metadata -eq 0 && $download_dataset -eq 0 ]] ; then
	print_help
	exit 0
fi

# Install python packages
#pip3 install numpy seaborn matplotlib scipy python-dotenv pydot librosa


# Download metadata
if [[ $download_metadata -eq 1 ]] ; then
	echo "Removing $metadata_file"
	rm -f "$metadata_file"
	wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
	unzip -o "$metadata_file"
fi

# Download dataset
if [[ $download_dataset -eq 1 ]] ; then
	echo "Removing $dataset_file"
	rm -f "$dataset_file"
	wget https://os.unil.cloud.switch.ch/fma/fma_small.zip
	unzip -o "$dataset_file"
fi

