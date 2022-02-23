#!/usr/bin/env bash

g++ src/bvp6c.cpp -O3 -msse2 -DNDEBUG -fopenmp -march=native -I/usr/local/include/eigen3 -I/usr/local/include/eigen3/unsupported -I/usr/local/include/boost -I./include -std=gnu++17 -lquadmath -o test/bvp6c
