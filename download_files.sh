#!/bin/bash

## make sure that you have your kaggle.json file in hand and that it is located in root!

# install kaggle api, I would advise doing this in a python venv 
pip install kaggle

while getopts d: flag
do
	case "${flag}" in
		d) dir=${OPTARG};;
	esac

done

kaggle competitions download m5-forecasting-uncertainty -q










