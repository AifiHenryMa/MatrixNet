#ifndef NET_H
#define NET_H
#endif // NET_H
#pragma once
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace MN {
    class Net {
    private:
        std::vector<int> layer_neuron_num; // 每一层神经元数目
        std::vector<cv::Mat> layer; // 层
        std::vector<cv::Mat> weights; // 权值矩阵
        std::vector<cv::Mat> bias; // 偏置项

    public:
        Net(); // 构造函数
        virtual ~Net(); // 虚析构函数
        // Initialize net:generate weights matrices,layer matrices, and bias matrices
        // bias default all zero
        void initNet(std::vector<int> layer_neuron_num_); // 用来初始化神经网络

        void initWeights(int type = 0, double a = 0., double b = 0.1); // 初始化权值矩阵，调用initWeight函数

        void initBias(cv::Scalar& bias); // 初始化偏置矩阵

        void Forward(); // 执行前向运算，包括线性运算和非线性激活，同时计算误差

        void Backword(); // 反向传播，调用updateWeights()函数更新权值

    protected:
        void initWeight(); // 初始化权值矩阵 if type = 0

        cv::Mat activationFunction();

        void deleleError();

        void updateWeights();

    };
}


