/**
*Author :Tharindra Galahena
*Project:implementation of little red riding hood problem in neural networks
*Date   :30/08/2012
*/

#ifndef NEURAL_NET_H_
#define NEURAL_NET_H_

#include <cstdlib>
#include <math.h>
#include <iostream>
#include <cstdlib>

typedef struct node{
	int num_inputs;
	float *weights;
	float *inputs;
	float *errors;
	float output;
} node;

typedef struct layer{
	int num_nodes;
	node *chr;
} layer;

using namespace std;

class neural{
	private:
		int num_inputs;
		int num_outputs;
		int num_layers;
		int num_weights;
		int num_hid_nodes;
		float leaning_rate;
		float momen;
		layer *layers;
		float *weights;
		
	public:
		neural();
		neural(int in, int out, int num, int hn, float lrate, float mom);		
		void init();	
		int get_num_weights(); 				
		float *get_weights();  				
		void put_weights(float *weights); 	
		float* feed(float *inputs);	
		float get_weighted_error(int l, int in);
		void learn(float *dout);	
		float convert(float input);	
		~neural();	
};

#endif
