CXX=g++
CXXFLAGS=-std=c++11 -O0
NVCC=nvcc


all : test_cuda1 test_cuda2

test_cuda1 : wholesale_cust_test.cu layer.cu matrix.cu utils/dataset.cpp
	${NVCC} ${CXXFLAGS} -o wholesale_cust_test wholesale_cust_test.cu layer.cu matrix.cu utils/dataset.cpp

test_cuda2 : large_input_test.cu layer.cu matrix.cu utils/dataset.cpp
	${NVCC} ${CXXFLAGS} -o large_input_test large_input_test.cu layer.cu matrix.cu utils/dataset.cpp

clean:
	-rm wholesale_cust_test large_input_test

