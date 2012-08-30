#
#Author :Tharindra Galahena
#Project:langton's ant simulator openGL and c++
#Date   :22/08/2012
#

all:
	g++ -c neural_net.cpp
	g++ -o main main.cpp neural_net.o 
