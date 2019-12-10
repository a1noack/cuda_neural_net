# 631_project README
###### Repository for class project for CIS631 Parallel Processing, Fall 2019, University of Oregon
Trevor Bergstrom & Adam Noack

### 631_project/Code
This directory contains all of the C++ code for our project. In this directory are four directories. The purpose of each will be explained in the following paragraphs.

##### 631_project/Code/1_non_matrix_serial_network

##### 631_project/Code/2_serial_matrix_network
This directory contains the code for our baseline implementation of the neural network that runs entirely on the CPU. The `matrix` class represents matrices in row major format and was constructed for easy passing of data between the host and the device.

##### 631_project/Code/3_cuda_matrix_network
This directory holds the code for our implementation of a neural network written using CUDA. In this implementation, each portion of our training setup (the dataset and each minibatch during training, the weight and bias matrices, and the loss gradients for each weight matrix) is represented as a `matrix` object. CUDA kernels were then written that operate on these row major matrices. These CUDA kernels and their wrapper functions that accept `matrix` objects can be found in `cuda_kernels.cu`. 

##### 631_project/Code/data
This directory holds our data files, and a subdirectory that holds our python scripts for generating testing datasets.

Each of our data files takes the following form:
  - the file header holds `n`, the number of samples in the dataset, followed by `m`, the number of dimensions of each sample, i.e. `#n,m`
  - each line holds one sample's values, separated by commas
  - the last value in each line is the number of the class to which the sample belongs
