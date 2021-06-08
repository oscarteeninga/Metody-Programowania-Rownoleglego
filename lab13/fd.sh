nvcc -I cuda-samples/Common/ lab3/FD_2D_global.cu -o global
nvcc -I cuda-samples/Common/ lab3/FD_2D_shared.cu -o shared
nvcc -I cuda-samples/Common/ lab3/FD_2D_texture_pad.cu -o texture
