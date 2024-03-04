#!/usr/bin/env bash

wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/evaluation/images.zip
mkdir -p data
mkdir -p data/splats
mkdir -p data/images
unzip tandt_db.zip -d data/splats
unzip images.zip -d data/images