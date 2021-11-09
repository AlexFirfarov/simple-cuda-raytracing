all:
	nvcc -ccbin=mpic++ -std=c++11 -Xcompiler -fopenmp main.cu
