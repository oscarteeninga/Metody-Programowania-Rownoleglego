#!/bin/bash
nvcc -I cuda-samples/Common/ cuda-samples/Samples/deviceQuery/deviceQuery.cpp -o dq
