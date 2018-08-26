#include <iostream>
#include "../include/Net.h"
#include <opencv2/core/core.hpp>
#include <stdio.h>
using namespace std;
using namespace cv;
using namespace mn;

int main(int argc, char* argv[]) {
    // set neuron number of every layer
    vector<int> layer_neuron_num = {784, 100, 10};
    // Initialize Nets and weights
    mn::Net net;
    net.initNet(layer_neuron_num);
    net.initWeights(0, 0, 0.01);
    net.initBias(0.05);
    cout << "Hello world!" << endl;
    getchar();
    return 0;
}
