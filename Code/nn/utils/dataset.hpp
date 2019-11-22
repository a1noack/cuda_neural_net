#ifndef dataset_hpp
#define dataset_hpp

#include <string>

class Dataset {
public:
    Dataset(char*, int);
    ~Dataset();
    void load_data(char*);
    void load_next_batch();
    void shuffle_sample_order();
    int get_batch_size();
    int n;
    float **batch_x;
    float **batch_y;
    int *get_sample_order();
private:
    std::string fname;
    int m;
    int k;
    float **x;
    float **y;
    int batch_size;
    int position;
    int *sample_order;
};

#endif /* dataset_hpp */
