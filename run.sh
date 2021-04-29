#!/bin/bash
if [ ! -d ./tsimd ]
then
	git clone https://github.com/jeffamstutz/tsimd
fi
g++ kernel_benchmark.cpp -o kernel_test -std=c++14 -march=native -I./tsimd/
if [ -e ./kernel_test ]
then
  ./kernel_test
fi
