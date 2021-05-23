#!/bin/bash
nvcc -I cuda-samples/Common/ cuda-samples/Samples/bandwidthTest/bandwidthTest.cu -o bt
