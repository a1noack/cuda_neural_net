# 631_project README
##### Repository for class project for CIS631 Parallel Processing, Fall 2019, University of Oregon
__Trevor Bergstrom & Adam Noack__

### 631_project/Code
This directory contains all of the C++ code for our project. In this directory are three subdirectories. The purpose of each will be explained in the following three subsections.
##### 631_project/Code/1_serial_network
This directory contains the code for our baseline implementation of the neural network that runs entirely on the CPU. The `matrix` class represents matrices in row major format and was constructed for easy passing of data between the host and the device.

Two tests are queued up to run. One runs on a artificial dataset, the other in the below mentioned Wholesale Customers dataset. Test results show the network layout, error, and time it took to run.

To run the tests, use make to build targets. Then run the 'run.batch' script to run on talapas node.
##### 631_project/Code/2_cuda_network
This directory holds the code for our implementation of a neural network written using CUDA. In this implementation, each portion of our training setup (the dataset and each minibatch during training, the weight and bias matrices, and the loss gradients for each weight matrix, etc.) is represented as a `matrix` object. CUDA kernels were then written that operate on these row major matrices. These CUDA kernels and their wrapper functions that accept `matrix` objects can be found in `cuda_kernels.cu`.

Two separate executables demonstrating the accuracy and speed of this implementation have been created: `wholesale_cust_test` and `large_input_test`. `wholesale_cust_test` shows that the MSE drops to a low level for the Wholesale Customers Dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wholesale+customers), indicating that our CUDA implementation can learn to approximate complex functions, as it should. `large_input_test` displays our CUDA net's ability to quickly work through samples with large input dimensionality by engaging in data parallelism. When given the exact same task and hyperparameters, the CUDA network finishes 100 epochs of training significantly faster than the serial network.

Two batch files have been written to run these executables.

##### 631_project/Code/data
This directory holds our data files and a subdirectory that holds our python scripts for generating testing datasets.

Each of our data files takes the following form:
- the file header holds `n`, the number of samples in the dataset, followed by `m`, the number of dimensions of each sample, i.e. `#n,m`
- each line holds one sample's values, separated by commas
- the last value in each line is the number of the class to which the sample belongs
