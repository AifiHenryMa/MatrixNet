#include <iostream>
#include "../include/Net.h"
#include <opencv2/core/core.hpp>
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

    // Get test samples and test samples
    Mat input, label, test_input, test_label;
    int sample_number = 800;
    net.get_input_label("data/input_label_1000.xml", input, label, sample_number, 0);
    net.get_input_label("data/input_label_1000.xml", test_input, test_label, 200, 800);

    // Set loss threshold, learning rate and activation function
    double loss_threshold = 0.5;
    net.learning_rate = 0.3;
    net.output_interval = 2;
    net.activation_function = "sigmoid";

    // Train, and draw the loss curve(cause the last parameter is true) and test the trained net
    net.train(input, label, loss_threshold, true);
    net.test(test_input, test_label);

    // Save the model
    net.save("models/model_sigmoid_800_200.xml");


    cout << "Hello world!" << endl;
    getchar();
    return 0;
}
