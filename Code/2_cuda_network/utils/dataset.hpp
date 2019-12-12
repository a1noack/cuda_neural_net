#ifndef dataset_hpp
#define dataset_hpp

#include <string>
/* The data set class handles loading the trining data from the file and making it accesible. Holds critical conditioning functions */
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
