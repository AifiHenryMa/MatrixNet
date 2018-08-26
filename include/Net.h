#ifndef NET_H
#define NET_H
#endif // NET_H
#pragma once
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace mn {
    class Net {
    private:
        std::vector<int> layer_neuron_num; // 每一层神经元数目
        std::vector<cv::Mat> layer; // 层
        std::vector<cv::Mat> weights; // 权值矩阵
        std::vector<cv::Mat> bias; // 偏置项
        std::vector<cv::Mat> delta_err;

        cv::Mat output_error;
        cv::Mat target;
        cv::Mat board;
        double loss;
        std::string activation_function = "sigmoid"; // 激活函数类型
        int output_interval = 10;
        double learning_rate = 0.1;
        double accuracy = 0.;
        std::vector<double> loss_vec;
        double fine_tune_factor = 1.01;

    public:
        Net(); // 构造函数
        virtual ~Net(); // 虚析构函数
        // Initialize net:generate weights matrices,layer matrices, and bias matrices
        // bias default all zero
        void initNet(std::vector<int> layer_neuron_num_); // 用来初始化神经网络
        void initWeights(int type = 0, double a = 0., double b = 0.1); // 初始化权值矩阵，调用initWeight函数
        void initBias(double bias_); // 初始化偏置矩阵
        void forWard(); // 执行前向运算，包括线性运算和非线性激活，同时计算误差
        void backWord(); // 反向传播，调用updateWeights()函数更新权值
        void calcLoss(cv::Mat& output, cv::Mat& target, cv::Mat& output_error, double& loss); // Objective function

    private:
        cv::Mat sigmoid(cv::Mat &x); // sigmoid function
        cv::Mat tanh(cv::Mat &x); // Tanh function
        cv::Mat ReLU(cv::Mat &x); // ReLU function
        cv::Mat derivativeFunction(cv::Mat& fx, std::string func_type); // Derivative function
        void initWeight(cv::Mat& dst, int type, double a, double b); // 初始化权值矩阵 if type = 0, Gaussian. else uniform.
        cv::Mat activationFunction(cv::Mat& x, std::string func_type); // 激活函数
        void deltaError(); // 计算 delta error
        void updateWeights(); // 更新权重
    };
}


