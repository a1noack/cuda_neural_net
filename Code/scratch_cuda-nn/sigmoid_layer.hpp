#include "layer.hpp"

class Sigmoid_Layer: public Layer {
public:
	
	Sigmoid_Layer(int);
	void forward_pass(Layer*);
	void back_prop_input(float*);
	void back_prop();
	~Sigmoid_Layer();

};