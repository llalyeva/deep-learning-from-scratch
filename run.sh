#!/bin/bash

echo "Adding some modules..."

module add gcc-10.2

echo "#################"
echo "    COMPILING    "
echo "#################"


## dont forget to use comiler optimizations (e.g. -O3 or -Ofast)
g++ -std=c++17 -march=native -O3 -ffast-math -funroll-loops -I src -I src/algebra -I src/dataset -I src/layers -I src/loss -I src/model src/*.cpp src/algebra/*.cpp src/dataset/*.cpp src/layers/*.cpp src/loss/*.cpp src/model/*.cpp -o network

echo "#################"
echo "     RUNNING     "
echo "#################"

## use nice to decrease priority in order to comply with aisa rules
## https://www.fi.muni.cz/tech/unix/computation.html.en
## especially if you are using multiple cores
nice -n 19 ./network
