CXX = g++
CXXFLAGS = -std=c++11 -Wall -O0

all: test_serial1 test_serial2

test_serial1 : serial_test.cpp layer.cpp matrix.cpp utils/dataset.cpp
	${CXX} ${CXXFLAGS} -o test_serial1 serial_test.cpp layer.cpp matrix.cpp utils/dataset.cpp

test_serial2 : serial_test2.cpp layer.cpp matrix.cpp utils/dataset.cpp
	${CXX} ${CXXFLAGS} -o test_serial2 serial_test2.cpp layer.cpp matrix.cpp utils/dataset.cpp

clean:
	-rm test_serial1 test_serial2
