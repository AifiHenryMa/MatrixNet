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

    // Update weights
    void Net::updateWeights() {
        for (unsigned int i = 0; i < weights.size(); ++i) {
            cv::Mat delta_weights = learning_rate * (delta_err[i]*layer[i].t());
            weights[i] = weights[i] + delta_weights;
        }
    }

    // backward
    void Net::backWard() {
        deltaError();
        updateWeights();
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

    // Draw loss curve
    void Net::draw_curve(cv::Mat& board, std::vector<double> points) {
        cv::Mat board_(620, 1000, CV_8UC3, cv::Scalar::all(200));
        board = board_;
        cv::line(board, cv::Point(0, 550), cv::Point(1000, 550), cv::Scalar(0, 0, 0), 2);
        cv::line(board, cv::Point(50, 0), cv::Point(50, 1000), cv::Scalar(0, 0, 0), 2);

        for (size_t i = 0; i < points.size() - 1; i++) {
            cv::Point pt1(50 + i * 2, (int)(548 - points[i]));
            cv::Point pt2(50 + i * 2 + 1, (int)(548 - points[i + 1]));
            cv::line(board, pt1, pt2, cv::Scalar(0, 0, 255), 2);
            if (i >= 1000)
                return;
        }
        cv::imshow("Loss", board);
        cv::waitKey(1);
    }

    void Net::train(const cv::Mat& input, const cv::Mat& target_, double accuracy_threshold) {
        if (input.empty()) {
            std::cout << "Input is empty!" << std::endl;
            return;
        }

        std::cout << "Train, begin!" << std::endl;

        cv::Mat sample;
        if (input.rows == (layer[0].rows) && input.cols == 1) {
            target = target_;
            sample = input;
            layer[0] = sample;
            forWard();

            int num_of_train = 0;
            while (accuracy < accuracy_threshold) {
                backWard();
                forWard();
                num_of_train++;
                if (num_of_train % 500 ==0) {
                    std::cout << "Train: " << num_of_train << "times" << std::endl;
                    std::cout << "Loss: " << loss << std::endl;
                }
            }

            std::cout << std::endl << "Train " << num_of_train << "times" << std::endl;
            std::cout << "Loss: " << loss << std::endl;
        }
        else if (input.rows == (layer[0].rows) && input.cols > 1) {
            double batch_loss = 0;
            int epoch = 0;
            while (accuracy < accuracy_threshold) {
                batch_loss = 0.;
                for (int i =0; i < input.cols; ++i) {
                    target = target_.col(i);
                    sample = input.col(i);

                    layer[0] = sample;
                    forWard();
                    batch_loss += loss;
                    backWard();
                }
                test(input, target_);
                epoch++;
                if (epoch % 10 == 0) {
                    std::cout << "Number of epoch: " << std::endl;
                    std::cout << "Loss sum: " << std::endl;
                }
            }
            std::cout << std::endl << "Number of epoch: " << epoch << std::endl;
            std::cout << "Loss sum: " << batch_loss << std::endl;
            std::cout << "Train successfully!" << std::endl;
        }
        else {
            std::cout << "Rows of input don't cv::Match the number of inputs!" << std::endl;
        }

    }

    // Train, use loss_threshold
    void Net::train(const cv::Mat& input, const cv::Mat& target_, double loss_threshold, bool draw_loss_curve) {
        if (input.empty()) {
            std::cout << "Input is empty!" << std::endl;
            return;
        }

        std::cout << "Train, begin!" << std::endl;

        cv::Mat sample;
        if (input.rows == (layer[0].rows) && input.cols == 1) {
            target = target_;
            sample = input;
            layer[0] = sample;
            forWard();
            int num_of_train = 0;
            while (loss > loss_threshold) {
                backWard();
                forWard();
                num_of_train++;
                if (num_of_train % 500 == 0) {
                    std::cout << "Train " << num_of_train << "times" << std::endl;
                    std::cout << "Loss: " << loss << std::endl;
                }
            }
            std::cout << std::endl << "Train " << num_of_train << " times" << std::endl;
            std::cout << "Loss: " << loss << std::endl;
            std::cout << "Train successfully!" << std::endl;
        }
        else if (input.rows == (layer[0].rows) && input.cols > 1) {
            double batch_loss = loss_threshold + 0.01;
            int epoch = 0;
            while (batch_loss > loss_threshold) {
                batch_loss = 0;
                for (int i = 0; i < input.cols; ++i) {
                    target = target_.col(i);
                    sample = input.col(i);
                    layer[0] = sample;

                    forWard();
                    backWard();

                    batch_loss += loss;
                }
                loss_vec.push_back(batch_loss);

                if (loss_vec.size() >=2 && draw_loss_curve)
                    draw_curve(board, loss_vec);
                epoch++;

                if (epoch % output_interval == 0) {
                    std::cout << "Number of epoch: " << epoch << std::endl;
                    std::cout << "Loss sum: " << batch_loss << std::endl;
                }
                if (epoch % 100 == 0) {
                    learning_rate *= fine_tune_factor;
                }
            }
            std::cout << std::endl << "Number of epoch: " << epoch << std::endl;
            std::cout << "Loss sum: " << batch_loss << std::endl;
            std::cout << "Train successfully!" << std::endl;
        }
        else {
            std::cout << "Rows of input don't cv::Match the number of input!" << std::endl;
        }
    }

    // Test
    void Net::test(const cv::Mat& input, const cv::Mat &target_) {
        if (input.empty()) {
            std::cout << "Input is empty!" << std::endl;
            return;
        }
        std::cout << std::endl << "Predict, begin!" << std::endl;

        if (input.rows == (layer[0].rows) && input.cols == 1) {
            int predict_number = predict_one(input);

            cv::Point target_maxLoc;

            cv::minMaxLoc(target_, NULL, NULL, NULL, &target_maxLoc, cv::noArray());
            int target_number = target_maxLoc.y;

            std::cout << "Predict: " << predict_number << std::endl;
            std::cout << "Target: " << target_number << std::endl;
            std::cout << "Loss: " << loss << std::endl;
        }
        else if (input.rows == (layer[0].rows) && input.cols > 1) {
            double  loss_sum = 0;
            int right_num = 0;
            cv::Mat sample;
            for (int i = 0; i < input.cols; ++i) {
                sample = input.col(i);
                int predict_number = predict_one(sample);
                loss_sum += loss;

                target = target_.col(i);
                cv::Point target_maxLoc;
                cv::minMaxLoc(target, NULL, NULL, NULL, &target_maxLoc, cv::noArray());
                int target_number = target_maxLoc.y;

                std::cout << "Test sample: " << i << "  " << "Predict: " << predict_number << std::endl;
                std::cout << "Test sample: " << i << "  " << "Target: " << target_number << std::endl << std::endl;
                if (predict_number == target_number) {
                    right_num++;
                }
            }
            accuracy = (double)right_num / input.cols;
            std::cout << "Loss sum: " << loss_sum << std::endl;
            std::cout << "accuracy: " << accuracy << std::endl;
        }
        else {
            std::cout << "Rows of input don't cv::Match the number of input!" << std::endl;
            return;
        }
    }

    // Predict
    int Net::predict_one(const cv::Mat& input) {
        if (input.empty()) {
            std::cout << "Input is empty!" <<std::endl;
            return -1;
        }
        else if (input.rows == (layer[0].rows) && input.cols ==1) {
            layer[0] = input;
            forWard();

            cv::Mat layer_out = layer[layer.size() -1];
            cv::Point predict_maxLoc;

            cv::minMaxLoc(layer_out, NULL, NULL, NULL, &predict_maxLoc, cv::noArray());
            return predict_maxLoc.y;
        }
        else {
            std::cout << "Please give one sample alone and ensure input.rows = layer[0].rows" << std::endl;
            return -1;
        }
    }

    // Predict, more than one sample
    std::vector<int> Net::predict(cv::Mat& input) {
        cv::Mat sample;
        std::vector<int> predicted_labels;
        if (input.rows == (layer[0].rows) && input.cols > 1) {
            for (int i = 0; i < input.cols; ++i) {
                sample = input.col(i);
                int predicted_label = predict_one(sample);
                predicted_labels.push_back(predicted_label);
            }
        }
        return predicted_labels;
    }

    // Get sample_number samples in XML file, from the start column.
    void Net::get_input_label(std::string filename, cv::Mat& input, cv::Mat& label, int sample_num, int start) {
        cv::FileStorage fs;
        fs.open(filename, cv::FileStorage::READ);
        cv::Mat input_, target_;
        fs["input"] >> input_;
        fs["target"] >> target_;
        fs.release();
        input = input_(cv::Rect(start, 0, sample_num, input_.rows));
        label = target_(cv::Rect(start, 0, sample_num, target_.rows));
    }

    // Save model;
    void Net::save(std::string filename) {
        cv::FileStorage model(filename, cv::FileStorage::WRITE);
        model << "layer_neuron_num" << layer_neuron_num;
        model << "learning_rate" << learning_rate;
        model << "activation_function" << activation_function;

        for (unsigned int i = 0; i < weights.size(); i++) {
            std::string weight_name = "weight_" + std::to_string(i);
            model << weight_name << weights[i];
        }
        model.release();
    }

    // Load model;
    void Net::load(std::string filename) {
        cv::FileStorage fs;
        fs.open(filename, cv::FileStorage::READ);
        cv::Mat input_, target_;

        fs["layer_neuron_num"] >> layer_neuron_num;
        initNet(layer_neuron_num);

        for (unsigned int i = 0; i < weights.size(); i++) {
            std::string weight_name = "weight_" + std::to_string(i);
            fs[weight_name] >> weights[i];
        }

        fs["learning_rate"] >> learning_rate;
        fs["activation_function"] >> activation_function;

        fs.release();
    }

}


