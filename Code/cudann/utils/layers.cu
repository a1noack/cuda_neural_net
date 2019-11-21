#include "layers.hpp"

/************************************
 * Put CUDA kernels here
 ***********************************/

//This is a dot product kernel takes two matrix data types and preforms dot product as if they were vectors. the result is stored in res
/*__global__ void dot_prod(float* a, float* b, float* res) {
    printf("%f, %f\n", a[0], b[0]);
    printf("%f, %f\n", a[1], b[1]);
    printf("%f, %f\n", a[2], b[2]);
    printf("%f, %f\n", a[3], b[3]);
    __shared__ int prod[MAX_THREADS_PER_BLOCK];
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    printf("tid: %d, blkdim: %d, blkidx: %d, tidx: %d\n", tid, blockDim.x, blockIdx.x, threadIdx.x);
    prod[threadIdx.x] = a[tid] * b[tid];
    printf("atid: %d, btid: %d, tid: %d\n", a[tid], b[tid], threadIdx.x);
    __syncthreads();

    if(threadIdx.x == 0) {
        int sum = 0;
        for(int i = 0; i < MAX_THREADS_PER_BLOCK; i++) {
            sum = sum + prod[i];
        }
        atomicAdd(res, sum);
    }
}*/
__global__ void dot_prod(float* a, float* b, float* res) {
    printf("%f, %f\n", a[0], b[0]);
    printf("%f, %f\n", a[1], b[1]);
    printf("%f, %f\n", a[2], b[2]);
    printf("%f, %f\n", a[3], b[3]);
    __shared__ int prod[MAX_THREADS_PER_BLOCK];
    prod[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
    printf("%d\n", threadIdx.x);
    __syncthreads();

    if(threadIdx.x == 0) {
        int sum = 0;
        for(int i = MAX_THREADS_PER_BLOCK - 1; i >= 0; i--) {
            sum = sum + prod[i];
        }
        *res = sum;
    }
}
__global__ void Linear_Forward_Pass(float* W, float* O, float* Z, float* b, int W_dim_x,
                                                                            int W_dim_y,
                                                                            int O_dim_x,
                                                                            int O_dim_y) {

    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col_idx = blockIdx.y * blockDim.y + threadIdx.y;

    int Z_dim_y = O_dim_y;
    int Z_dim_x = W_dim_x;

    float Z_sum = 0;

    if(row_idx < Z_dim_x && col_idx < Z_dim_y) {
       for(int i = 0; i < W_dim_y; i++) {
            printf("W: %f, O: %f\n", W[row_idx * W_dim_y + i], O[i * O_dim_x + col_idx]);
           Z_sum += W[row_idx * W_dim_y + i] * O[i * O_dim_x + col_idx];
       }
       Z[row_idx * Z_dim_y + col_idx] = Z_sum + b[row_idx];
    }
}



/***************************************
 * Class function implementations here
 ***************************************/

// Parent class

void Layer::connect(Layer* prev) {
    this->previous = prev;
    this->previous->set_next(this);
}

void Layer::set_next(Layer* nxt) {
    this->next = nxt;
}

void Layer::init_weights() {

    this->outputs = new matrix();
    this->bias = new matrix();
    this->dBias = new matrix();

    outputs->set_mem_zero(1,num_nodes);
    bias->set_mem_zero(1, num_nodes);
    dBias->set_mem_zero(1, num_nodes);

    if(this->previous != NULL) {
        int weight_count = this->previous->get_num_nodes();
        weight_count *= this->num_nodes;

        weights = new matrix();
        dWeights = new matrix();
        weights-> set_mem_random(1, weight_count);
        dWeights-> set_mem_random(1, weight_count);
    }
}

// TESTING FUNCTIONS

void Layer::set_weights(float* a, int x, int y) {
    this->weights = new matrix();
    this->weights->set_mem(a, x, y);
}

void Layer::set_bias(float* a, int x, int y) {
    this->bias = new matrix();
    this->bias->set_mem(a, x, y);
}
void Layer::set_outs(float* a, int x, int y) {
    this->outputs = new matrix();
    this->outputs->set_mem(a, x, y);
}

void Layer::print_outs() {
    outputs->print();
}

void Layer::print_weights() {
    weights->print();
}

void Layer::print_bias() {
    bias->print();
}
/*********************************************************
 * Linear Layer
 ********************************************************/

Linear_Layer::Linear_Layer(char* n_name, int nodes_num) {
    this->num_nodes = nodes_num;
    this->name = n_name;
    this->previous = NULL;
    this->next = NULL;
}

//void Linear_Layer::alt_forward() {
    //int num_blocks = ceil( float(W_num_cols) / MAX_THREADS_PER_BLOCK);
    //Linear_Forward_Pass<<<num_blocks, MAX_THREADS_PER_BLOCK>>>(this->weights->host_data

void Linear_Layer::forward() {
    matrix* prev_outs = this->previous->get_outputs();
    int W_num_cols = this->weights->dim_y;
    int W_num_rows = this->weights->dim_x;

    int O_num_rows = prev_outs->dim_x;
    int O_num_cols = prev_outs->dim_y;

    float* outs = new float[W_num_rows * O_num_cols];
    int outs_itr = 0;


    //need to check dimensions. The y of the weights dimension should be the x of the outputs

    int num_blocks = ceil( float(W_num_cols) / MAX_THREADS_PER_BLOCK);
    printf("we got shit\n");

    for(int i = 0; i < O_num_cols; i++) {
        printf("iterating i=%d\n", i);
        float* col =prev_outs->get_col(i);
        float* dev_col;
        cudaMalloc(&dev_col, 4 * sizeof(float));
        cudaMemcpy(dev_col, col, 4 * sizeof(float), cudaMemcpyHostToDevice);
        //printf("COL: %f, %f, %f, %f\n", col[0], col[1], col[2], col[3]);
        for(int j = 0; j < W_num_rows; j++) {
                //printf("iterating j=%d\n", j);
                float* row = weights->get_row(j);
                float* dev_row;
                cudaMalloc(&dev_row, 4 * sizeof(float));
                cudaMemcpy(dev_row, row, 4 * sizeof(float), cudaMemcpyHostToDevice);
                //printf("ROW: %f, %f, %f, %f\n", row[0], row[1], row[2], row[3]);

                dot_prod<<<num_blocks, MAX_THREADS_PER_BLOCK>>>(dev_col, dev_row, &outs[outs_itr]);
            printf("just computed: %f\n", outs[outs_itr]);
            outs_itr++;
            delete row;
        }
        delete col;
    }
    printf("done getting dots\n");
    printf("%f, %f, %f\n", outs[0], outs[1], outs[2]);
    this->outputs->set_mem(outs, W_num_rows, O_num_cols);
    delete outs;
}

void Linear_Layer::backprop() {
}

/*********************************************************
 * RELU Layer
 ********************************************************/

RELU_Layer::RELU_Layer(char* n_name, int nodes_num) {
    num_nodes = nodes_num;
    name = n_name;
}

void RELU_Layer::forward() {
}

void RELU_Layer::backprop() { }
/*********************************************************
 * SIGMOID Layer
 ********************************************************/

Sigmoid_Layer::Sigmoid_Layer(char* n_name, int nodes_num) {
    num_nodes = nodes_num;
    name = n_name;
}

void Sigmoid_Layer::forward() { }

void Sigmoid_Layer::backprop() { }


