/**
*Author :Tharindra Galahena
*Project:implementation of little red riding hood problem in neural networks
*Date   :30/08/2012
*/

#include <cstdlib>
#include <math.h>
#include <stdio.h>
#include <iostream>

#include "neural_net.h"

#define		NUM_INPUTS			6
#define		NUM_OUTPUTS			7
#define		NUM_CHARACTORS		3
#define		NUM_LAYERS			3
#define		NUM_HIDDEN_NODES	5
#define		NUM_ITARETIONS	10000
#define		ERROR_TOLERENCE	 0.01
#define 	LEANING_RATE	 0.20
#define 	MOMENTOM		  0.8	

string charactors[NUM_CHARACTORS] = {
				"wolf",
				"grand mother",
				"Woodcutter"
			};
string questions[NUM_INPUTS] = {
				
				"has big years ? (y/n) : ",
				"has big eyes ? (y/n)  : ",
				"has big teeth ? (y/n) : ",
				"is kindly ? (y/n)     : ",
				"is winkles ? (y/n)    : ",
				"is handsome ? (y/n)   : "
			};
			
string response[NUM_OUTPUTS] = {
				
				"run away!",
				"screem!",
				"look for woodcutter!",
				"aproche!",
				"kiss on cheeks!",
				"offer food!",
				"flirt with!"
			};
/*
 input
 */
float test[NUM_CHARACTORS][NUM_INPUTS] = {
		{1.0, 1.0, 1.0, 0.0, 0.0, 0.0}, // wolf
		{0.0, 1.0, 0.0, 1.0, 1.0, 0.0}, // grand mother
		{1.0, 0.0, 0.0, 1.0, 0.0, 1.0}  // woodcuttor
	};
	
/*
 outputs are run away, screem, look for wood cuttor, aproche, kiss in cheaks, 
 offer food and flirt with in this order.
*/    
float dout[NUM_CHARACTORS][NUM_OUTPUTS] = {
		{1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0}, // Wolf
		{0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0}, // Grand Mother
		{0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0}  // Woodcutter
	};

float get_out(float i){
	if(i > 0.5) return 1.0;
	else return 0.0;
}

void learn(neural *net){
	float ms[NUM_CHARACTORS];
	cout << endl;
	cout << "Little red riding hood is learning !!! " << endl;
	char c;
	lll: ;
	for(int j = 0;j < NUM_ITARETIONS; j++){
		for(int i = 0;i < NUM_CHARACTORS; i++){
			float *out = net -> feed(test[i]);
			cout << "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b";
			cout << "iterations : " << j;
			cout.flush();
			ms[i] = 0;
			for(int t = 0;t < 7; t++){
				ms[i] += pow(dout[i][t] - out[t], 2.0);
			}
			net -> learn(dout[i]);
		}
	}
	cout << endl;
	cout << "	-- DONE --	" << endl;
	cout << endl;
	for(int i = 0; i < 3; i++){
		cout << "  Error for " << charactors[i] << " = " << ms[i] << endl;
	}
	cout << endl;
	for(int i = 0; i < 3; i++){
		if(ms[i] > ERROR_TOLERENCE)
			cout << "I'm still not clear about " << charactors[i] << " !" << endl;
	}
	cout << endl;
	cout << " more leaning ? (y/n) : ";
	cin >> c;
	cout << endl;
	cout << endl;
	if(c == 'y') goto lll;
}
void go_home(neural *net){
	char c;
	float *out;
	float input[NUM_INPUTS];
	cout << endl;
	cout << "So the Little Red Riding Hood went to Grand Mother's House ...." << endl;
	cout << "..........And she show some one there ..." << endl;
	cout << endl;
	cout << "Questions : " << endl; 
	cout << endl;
	cout << "Does he/she/it has, " << endl;
	cout << endl;
	for(int i = 0; i < 6; i++){
		
			cout << "	" << questions[i];
			cin >> c;
			
			if(c == 'y') input[i] = 1.0;
			else input[i] = 0.0;
		}

	out = net -> feed(input);
	cout << endl;
	cout << "Little Red Riding Hood........ " << endl; 
	cout << endl;
	for(int i = 0; i < NUM_OUTPUTS; i++){
		if(get_out(out[i]) == 1.0)
			cout << response[i]  << ", ";
	}
	cout << endl;
	cout << endl;
	cout << endl;	
}
char menu(){
	char c;
	cout << " 	MENU " << endl;
	cout << "-------------------------------------" << endl;
	cout << "	1 - learn " << endl;
	cout << "	2 - go to grand mother's house " << endl;
	cout << "	3 - quit " << endl;
	cout << endl;
	cout << "Enter Number  : ";
	cin >> c;
	cout << endl;
	return c;
}
int main(){
	 		
	srand(time(0));
	neural *net = new neural( 
								NUM_INPUTS, 
								NUM_OUTPUTS, 
								NUM_LAYERS, 
								NUM_HIDDEN_NODES, 
								LEANING_RATE, 
								MOMENTOM
							);
	net -> init();
	
	cout << "----------------------------------------------------------------\n" << endl;
	cout << "        Little Red Riding Hood Problem in Nural Networks          " << endl;
	cout << "       --------------------------------------------------       \n" << endl;
	cout << " * Created By : Tharindra Galahena ( inf0_warri0r )               " << endl;
	cout << " * Blog       : http://www.inf0warri0r.blogspot.com               " << endl;
	cout << "----------------------------------------------------------------\n" << endl;
	cout << endl;
	while(1){
		char c = menu();
		if(c == '1')
			learn(net);
		else if(c == '2')
			go_home(net);
		else if(c == '3'){
			break;
		}
	}
	net -> ~neural();
	return 0;
}


