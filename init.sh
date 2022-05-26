#!/bin/bash
# Uncompress data files
tar -xvzf data.zip

# create necessary dirs
mkdir -p ckpt
mkdir -p out
mkdir -p tensorboard
