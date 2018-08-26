// Implement the function included in net.h
#include "../include/Net.h"

using namespace mn;

namespace mn {

    Net::Net()
    {
        //ctor
    }

    Net::~Net()
    {
        //dtor
    }

    // Initialize net
    void Net::initNet(std::vector<int> layer_neuron_num_) {
        layer_neuron_num = layer_neuron_num_;
        // 生成每一层
        layer.resize(layer_neuron_num_.size()); // 层数
        for (unsigned int i = 0; i < layer.size(); i++) {
            layer[i].create(layer_neuron_num[i], 1, CV_32FC1); //  create(nrows, ncols, type)
        }
        std::cout << "Generate layers, successfully!" << std::endl;

        // 生成每一个权重矩阵和偏置
        weights.resize(layer.size() - 1);
        bias.resize(layer.size() -1);
        for (unsigned int i = 0; i < (layer.size() - 1); ++i) {
            weights[i].create(layer[i + 1].rows, layer[i].rows, CV_32FC1);
            bias[i] = cv::Mat::zeros(layer[i+1].rows, 1, CV_32FC1);
        }
        std::cout << "Generate weights matrices and bias, successfully!" << std::endl;
        std::cout << "Initialise Net, done!" << std::endl;
    }

    // Initialize the weights matrix. if type = 0, Gaussian. else uniform.
    void Net::initWeight(cv::Mat &dst, int type, double a, double b) {
        if (type == 0) {
            randn(dst, a, b); // Fills the array with normally distributed random numbers.
        }
        else {
            randu(dst, a, b); // Generates a single uniformly-distributed random number or an array of random numbers.
        }
    }

    // Initialize the weights matrix.
    void Net::initWeights(int type, double a, double b) {
        // Initialize weights cv::Matrices and bias
        for (unsigned int i = 0; i < weights.size(); ++i) {
            initWeight(weights[i], 0, 0., 0.1);
        }
    }

    // Initialzie the bias matrices.
    void Net::initBias(double bias_) {
		for (unsigned int i = 0; i < bias.size(); i++) {
			bias[i] = bias_;
		}
	}

        // sigmoid function
    cv::Mat Net::sigmoid(cv::Mat& x) {
        cv::Mat exp_x_, fx;
        cv::exp(-x, exp_x_);
        fx = 1.0 / (1.0 + exp_x_);
        return fx;
    }

    // tanh function
    cv::Mat Net::tanh(cv::Mat& x) {
        cv::Mat exp_x_, exp_x, fx;
        cv::exp(-x, exp_x_);
        cv::exp(x, exp_x);
        fx = (exp_x - exp_x_) / (exp_x + exp_x_);
        return fx;
    }

    // ReLU function
    cv::Mat Net::ReLU(cv::Mat &x) {
        cv::Mat fx = x;
        for (int i = 0; i < fx.rows; i++) {
            for (int j = 0; j < fx.cols; j++) {
                if (fx.at<float>(i, j) < 0) {
                    fx.at<float>(i, j) = 0; // 用at可以直接访问到某个位置的像素
                }
            }

        }
        return fx;
    }

    // Derivation function
    cv::Mat Net::derivativeFunction(cv::Mat& fx, std::string func_type) {
        cv::Mat dx;
        if (func_type == "sigmoid")
            dx = sigmoid(fx).mul((1-sigmoid(fx)));
        if (func_type == "tanh") {
            cv::Mat tanh_2;
            cv::pow(tanh(fx), 2., tanh_2);
            dx = 1 - tanh_2;
        }
        if (func_type == "ReLU") {
            dx = fx;
            for (int i = 0; i < fx.rows; i++)
                for (int j = 0; i < fx.cols; j++)
                    if (fx.at<float>(i, j) > 0)
                        dx.at<float>(i, j) = 1;
        }
        return dx;
    }

    // Objective function
    void Net::calcLoss(cv::Mat& output, cv::Mat& target, cv::Mat& output_error, double& loss) {
        if (target.empty()) {
            std::cout << "Can't find the target cv::Matrix" << std::endl;
            return;
        }

        output_error = target - output;
        cv::Mat err_square;
        cv::pow(output_error, 2., err_square);
        cv::Scalar err_sqr_sum = cv::sum(err_square);
        loss = err_sqr_sum[0] / (double)(output.rows);
    }

    // Forward
    void Net::forWard() {
        for (unsigned int i = 0; i < layer_neuron_num.size() - 1; ++i) {
            cv::Mat product = weights[i] * layer[i] + bias[i];
            layer[i+1] = activationFunction(product, activation_function);
        }
    }

    // Compute delta error
    void Net::deltaError() {
        delta_err.resize(layer.size() - 1);
        for (unsigned i = delta_err.size() -1; i>=0; i--) {
            delta_err[i].create(layer[i+1].size(), layer[i+1].type());
            cv::Mat dx = derivativeFunction(layer[i+1], activation_function);
            // Output layer delta error
            if (i == delta_err.size() -1) {
                delta_err[i] = dx.mul(output_error);
            }
            else { // Hidden layer delta error
                cv::Mat weight = weights[i];
                cv::Mat weight_t = weights[i].t();
                cv::Mat delta_err_1 = delta_err[i];
                delta_err[i] = dx.mul(weights[i+1].t() * delta_err[i+1]);
            }
        }
    }

    // Forward
    void Net::backWord() {
        deltaError();
        updateWeights();
    }

    // Update weights
    void Net::updateWeights() {
        for (unsigned int i = 0; i < weights.size(); ++i) {
            cv::Mat delta_weights = learning_rate * (delta_err[i]*layer[i].t());
            weights[i] = weights[i] + delta_weights;
        }
    }

    // Activation function
    cv::Mat Net::activationFunction(cv::Mat& x, std::string func_type) {
        activation_function = func_type;
        cv::Mat fx;
        if (func_type == "sigmoid")
            fx = sigmoid(x);
        else if (func_type == "tanh")
            fx = tanh(x);
        else if (func_type == "ReLU")
            fx = ReLU(x);
        else
            std::cout << "Error: without " << func_type << " function!" << std::endl;
        return fx;
    }

}


