#!/bin/bash

# Compute metrics for sinc interpolation
# There are two set of target sample rates: 16000 and 48000
# For 16000, we set the input sample rate [2000, 4000, 8000, 12000]
# For 48000, we set the input sample rate [8000, 12000, 16000, 24000]
# Example to run: python compute_metrics_sinc.py -sr 8000 -tsr 48000

# TARGET_SR=(16000 48000)
TARGET_SR=(48000)

# Loop over target sample rate
for tsr in ${TARGET_SR[@]}; do
    # If condition to check target sample rate
    if [ $TARGET_SR -eq 16000 ]; then
        INPUT_SR=(2000 4000 8000 12000)
    elif [ $TARGET_SR -eq 48000 ]; then
        INPUT_SR=(8000 12000 16000 24000)
    else
        echo "Invalid target sample rate"
        exit 1
    fi

    # Loop over input sample rate
    for sr in ${INPUT_SR[@]}; do
        python compute_metrics_sinc.py -sr $sr -tsr $tsr
    done
done