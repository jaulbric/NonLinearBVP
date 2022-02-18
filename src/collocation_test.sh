#!/usr/bin/env bash

g++ src/collocation_test.cpp -O3 -msse2 -DNDEBUG -fopenmp -march=native -I/usr/local/include/eigen3 -I/usr/local/include/eigen3/unsupported -I/usr/local/include/boost -I./include -std=c++17 -o test/collocation_test
