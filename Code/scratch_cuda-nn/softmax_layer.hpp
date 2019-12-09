#include "layer.hpp"

class Softmax_Layer: public Layer {
private:
	float* dp;
	float esum;
public:
	
	Softmax_Layer(int);
	void forward_pass(Layer*);
	void back_prop_input(float*);
	void back_prop();
	~Softmax_Layer();

};