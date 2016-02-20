CC=/usr/local/cuda-7.5/bin/nvcc -ccbin g++ -std=c++11 -w  
CUDADIR=/home/Documents/NVIDIA_CUDA-7.5_Samples/

all: cipher

cipher: cipher.o
	$(CC) -o $@ $<  -lcufft
cipher.o: cipher.cu
	$(CC) -o $@ -c $<
clean:
	rm *.o cipher