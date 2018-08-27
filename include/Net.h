#ifndef NET_H
#define NET_H
#endif // NET_H
#pragma once
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace mn {
    class Net {
    private:
        // 整数向量：指定每一层的神经元个数，包括输入和输出
        std::vector<int> layer_neuron_num; // 每一层神经元数目
        std::vector<cv::Mat> layer; // 层
        std::vector<cv::Mat> weights; // 权值矩阵
        std::vector<cv::Mat> bias; // 偏置项
        std::vector<cv::Mat> delta_err;

        cv::Mat output_error;
        cv::Mat target;
        cv::Mat board;
        double loss;
        double accuracy = 0.;
        std::vector<double> loss_vec; // 定义损失向量
        double fine_tune_factor = 1.01;

    public:
        Net();  // 构造函数
        ~Net(); // 析构函数
        // Initialize net:generate weights matrices,layer matrices, and bias matrices
        // bias default all zero
        void initNet(std::vector<int> layer_neuron_num_); // 用来初始化神经网络
        void initWeights(int type = 0, double a = 0., double b = 0.1); // 初始化权值矩阵，调用initWeight函数
        void initBias(double bias_); // 初始化偏置矩阵
        void forWard(); // 执行前向运算，包括线性运算和非线性激活，同时计算误差
        void backWard(); // 反向传播，调用updateWeights()函数更新权值
        void calcLoss(cv::Mat& output, cv::Mat& target, cv::Mat& output_error, double& loss); // Objective function
        double learning_rate = 0.1; // 定义学习率
        int output_interval = 10;
        std::string activation_function = "sigmoid"; // 激活函数类型


        // Train, use accuracy_threshold
        void train(const cv::Mat& input, const cv::Mat& target, double accuracy_threshold);

        // Train, use loss_threshold
        void train(const cv::Mat& input, const cv::Mat& target_, double loss_threshold, bool draw_loss_curve = false); // 训练函数

        // Test
        void test(const cv::Mat& input, const cv::Mat& target_);

        // Predict,just one sample
        int predict_one(const cv::Mat& input);

        // Predict, more than one sample
        std::vector<int> predict(cv::Mat& input);

        // Save model
        void save(std::string filename);

        // Load model
        void load(std::string filename);

        // Get sample_number samples in XML file, from the start column.
        void get_input_label(std::string filename, cv::Mat& input, cv::Mat& label, int sample_num, int start);

    private:
        cv::Mat sigmoid(cv::Mat &x); // sigmoid function
        cv::Mat tanh(cv::Mat &x); // Tanh function
        cv::Mat ReLU(cv::Mat &x); // ReLU function
        cv::Mat derivativeFunction(cv::Mat& fx, std::string func_type); // Derivative function
        void initWeight(cv::Mat& dst, int type, double a, double b); // 初始化权值矩阵 if type = 0, Gaussian. else uniform.
        cv::Mat activationFunction(cv::Mat& x, std::string func_type); // 激活函数
        void deltaError(); // 计算 delta error
        void updateWeights(); // 更新权重
        // Draw loss curve
        void draw_curve(cv::Mat& board, std::vector<double> points);
    };
}


