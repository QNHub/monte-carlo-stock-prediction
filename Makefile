CXX  = g++
NVCC = nvcc
FLAGS = -O2 -std=c++17

cpu: monte_carlo_cpu.cpp
	$(CXX) $(FLAGS) -o monte_carlo_cpu $

gpu: monte_carlo_gpu.cu
	$(NVCC) $(FLAGS) -o monte_carlo_gpu $

clean:
	rm -f monte_carlo_cpu monte_carlo_gpu
