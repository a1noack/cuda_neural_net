#include "dataset.hpp"
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <algorithm>

Dataset::Dataset(std::string fname, int batch_size) {
    this->load_data(fname);
    this->fname = fname;
    this->batch_size = batch_size;
    this->minibatch = new float*[this->batch_size];
    for(int i = 0; i < this->batch_size; i++)
        this->minibatch[i] = new float[this->m];
    this->sample_order = int[this->n];
    for(int i = 0; i < this->n; i++) this->sample_order[i] = i;
    this->shuffle_sample_order();
}

void Dataset::load_data(std::string fname) {
    // load file
    std::ifstream file(fname);
    
    // make sure file was loaded correctly
    if(!file.good()) {
        printf("file load failed");
        return;
    }

    std::string line;
    std::string delimeter = ",";

    // process header
    std::getline(file, line);
    this->n = stoi(line.substr(2, line.find(delimeter)));
    this->m = stoi(line.substr(line.find(delimeter) + 1));
    this->k = 2; 
    
    // allocate space on heap for data
    int r, c;
    this->x = new float*[this->n];
    this->y = new float*[this->n];
    for(r = 0; r < this->n; r++){
        this->x[r] = new float[this->m];
        this->y[r] = new float[this->m];
        for(int a = 0; a < this->k; a++) y[r][a] = 0;
    }

    // load data from file into allocated space
    std::string val;
    float label;
    for(r = 0; r < this->n; r++) {
        std::getline(file, line);
        if(!file.good())
            break;
        std::stringstream iss(line);
        for(c = 0; c < this->m + 1; c++) {
            std::getline(iss, val, ',');
            if(!file.good())
                break;
            std::stringstream convertor(val);
            if(c < this->m){
                convertor >> this->x[r][c];
//                printf("%f ", x[r][c]);
            } 
            else{
                convertor >> label;
                this->y[r][int(label)] = 1.0;
//                printf(" label: [%f, %f]\n", y[r][0], y[r][1]);
            }
        }
    }
    printf("x[1][5] = %f", this->x[1][5]);
}

void Dataset::load_next_batch() {
    for(int i = 0; i < this->batch_size; i++) {
        idx = this->sample_order[this->position + i];
        this->minibatch[i] = this->x[idx];
    }
    this->position += this->batch_size;
    if(this->position >= this->n - this->batch_size)
        this->position = 0;
}

void Dataset::shuffle_sample_order() {
   random_shuffle(&this->sample_order[0], &this->sample_order[n-1]);
}


