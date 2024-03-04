#!/usr/bin/env bash

wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
mkdir -p data
mkdir -p data/splats
unzip tandt_db.zip -d data/splats