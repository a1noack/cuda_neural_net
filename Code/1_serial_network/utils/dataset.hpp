#ifndef dataset_hpp
#define dataset_hpp

#include <string>

//This is the dataset class, it handles all the loading of the data from a file. Then preforms some crucial operations on it, conditioning it for network training.
class Dataset {
public:
    Dataset(const char*, int);
    ~Dataset();
    void load_data(const char*);
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
