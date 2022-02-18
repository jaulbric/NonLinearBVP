#!/usr/bin/env bash

g++ test/benchmark_test.cpp -O3 -msse2 -DNDEBUG -fopenmp -march=native -I/usr/local/include/eigen3 -I/usr/local/include/eigen3/unsupported -I./include -std=c++17 -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread -o test/benchmark_test
