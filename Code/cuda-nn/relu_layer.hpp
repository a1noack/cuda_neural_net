#include "layer.hpp"

class RELU_Layer: public Layer {
public:
	
	RELU_Layer(int);
	void forward_pass(Layer*);
	void back_prop_input(float*);
	void back_prop();
	~RELU_Layer();

};