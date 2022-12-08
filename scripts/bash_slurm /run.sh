#!/bin/bash
# THIS SCRIPT WILL BE IN THE DOCKER CONTAINER
# Here its a test script with timer for completion, with logging of output
#time python -u test.py >> output.txt &
#time python -u test.py

rm -rf classification/results/models/*
rm -rf classification/results/study/*
rm -rf classification/results/log/*

#time python classification/src/hpo_classification.py -ne 5 -nt 100 -stnm cl_ne_5_nt_100 -h5
