#!/bin/bash

# This script will install the required python packages and download the dataset and metadata.

print_help() {
	echo "Usage:"
	echo "$ $0  <options>"
	echo ""
	echo "  options:"
	echo "	-d - Download dataset"
	echo "	-m - Download metadata"
	echo "	-i - Install python packages"
	echo "	-h - This help string"
	echo ""
}

download_meta_data=0
download_dataset=0
install_packages=0
metadata_file=fma_metadata.zip
dataset_file=fma_small.zip
help=false


while getopts "hdmi" options; do
	case "${options}" in
		d)
			download_dataset=1
			echo "Downloading dataset"
			;;
		m)
			download_metadata=1
			;;
		i)
			install_packages=1
			;;

		h | *)
			print_help
			exit 0
			;;
	esac
done



if [[ $download_metadata -eq 0 && $download_dataset -eq 0 && $install_packages -eq 0 ]] ; then
	print_help
	exit 0
fi

# Install python packages
if [[ $install_packages -eq 1 ]] ; then
	echo "Installing packages"
	pip3 install numpy matplotlib librosa tqdm
fi


# Download metadata
if [[ $download_metadata -eq 1 ]] ; then
	echo "Downloading metadata"
	echo "Removing $metadata_file"
	rm -f "$metadata_file"
	wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
	unzip -o "$metadata_file"
fi

# Download dataset
if [[ $download_dataset -eq 1 ]] ; then
	echo "Downloading dataset"
	echo "Removing $dataset_file"
	rm -f "$dataset_file"
	wget https://os.unil.cloud.switch.ch/fma/fma_small.zip
	unzip -o "$dataset_file"
fi

