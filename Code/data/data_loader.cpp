#include "nn.hpp"
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <string>

void load_data(std::string fname, float **x, float **y, int *n, int *m) {
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
    int i = stoi(line.substr(2, line.find(delimeter)));
    n = &i;
    int j = stoi(line.substr(line.find(delimeter) + 1));
    m = &j; 

    int r, c;
    int k = 2;
    // process data
    x = (float **)malloc(*n * sizeof(float *));
    y = (float **)malloc(*n * sizeof(float *));
    for(r = 0; r < *n; r++) {
        x[r] = (float *)malloc(*m * sizeof(float));
        y[r] = (float *)malloc(k * sizeof(float));
        for(int a = 0; a < k; a++) y[r][a] = 0;
    }
    std::string val;
    float label;
    for(r = 0; r < *n; r++) {
        std::getline(file, line);
        if(!file.good())
            break;
        std::stringstream iss(line);
        for(c = 0; c < *m + 1; c++) {
            std::getline(iss, val, ',');
            if(!file.good())
                break;
            std::stringstream convertor(val);
            if(c < *m){
                convertor >> x[r][c];
                printf("%f ", x[r][c]);
            } 
            else{
                convertor >> label;
                y[r][int(label)] = 1.0;
                printf(" label: [%f, %f]\n", y[r][0], y[r][1]);
            }
        }
    }
}

int main() {
    std::string fname = "../data/data_n100_m5_mu1.5.csv";

    float **x, **y;
    int *n, *m;

    load_data(fname, x, y, n, m);
}

