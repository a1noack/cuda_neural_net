#ifndef dataset_hpp
#define dataset_hpp

#include <string>

class Dataset {
public:
    Dataset() = default;
    void load_data();
    void load_next_batch();
    void shuffle_sample_order();
    int n;
    float **x;
    float **y;
    float **minibatch;
private:
    std::string fname;
    int m;
    int k;
    int batch_size;
    int position;
    int sample_order[];
};

#endif /* dataset_hpp */
