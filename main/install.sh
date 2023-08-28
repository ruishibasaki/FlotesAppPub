#!/bin/bash

mkdir ../bin
unzip nantes_graph_.zip -d graphs
cd ../simulation 
make clean 
make
cd ../main/

#run
#python3 main.py ./inputs/noreal_CF_10_24_1.txt
