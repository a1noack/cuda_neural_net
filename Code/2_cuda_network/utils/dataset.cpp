#define _GLIBCXX_USE_CXX11_ABI 0
#include "dataset.hpp"
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>

//Constructor for the dataset, handles all of the initilaization and reading and loading data
Dataset::Dataset(const char* fname, int batch_size) {
    load_data(fname);
    this->fname = fname;
    this->batch_size = batch_size;

    batch_x = new float*[batch_size];
    batch_y = new float*[batch_size];
    sample_order = new int[n];
    for(int i = 0; i < n; i++) {
        sample_order[i] = i;
    }
position = 0;
}

//dataset destructor
Dataset::~Dataset() {
    delete batch_x;
    delete batch_y;

    delete sample_order;

    for(int r = 0; r < n; r++) {
        delete x[r];
        delete y[r];
    }
    delete x;
    delete y;
}

//Function loads all the data from the file to the class
void Dataset::load_data(const char* fname) {
    // load file
    std::ifstream file(fname);

    // make sure file was loaded correctly
    if(!file.good()) {
        printf("file load failed\n");
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
            }
            else{
                convertor >> label;
                this->y[r][int(label)] = 1.0;
            }
        }
    }
}

//Function loads the next batch into the correct access parameters
void Dataset::load_next_batch() {
    int idx;
    for(int i = 0; i < this->batch_size; i++) {
        idx = this->sample_order[this->position + i];
        this->batch_x[i] = this->x[idx];
        this->batch_y[i] = this->y[idx];
    }
    this->position += this->batch_size;
    if(this->position >= this->n - this->batch_size)
        this->position = 0;
}

//Important function to shuffle the sample order. Needs to happen every epoch
void Dataset::shuffle_sample_order() {
    std::random_shuffle(&this->sample_order[0], &this->sample_order[this->n]);
}

int Dataset::get_batch_size() {
    return this->batch_size;
}

int *Dataset::get_sample_order() {
    return this->sample_order;
}
