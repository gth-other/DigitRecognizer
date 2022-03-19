#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>


class NeuralNetwork {
public:
    NeuralNetwork(int l1_size, int l2_size, int l3_size, int l4_size);
    static int predict(const NeuralNetwork& nn, std::vector<float> l1);
    static NeuralNetwork train(NeuralNetwork nn, std::vector<std::vector<float>> data, std::vector<int> answers, float eta, float necessary_MSE, bool log);
    static void save(NeuralNetwork nn, const std::string& location);
    NeuralNetwork(const char* location);
private:
    static float _sigmoid(float number);
    static float _sigmoid_derivative(float number);
    static std::vector<float> _calc_input_signals(std::vector<float> l_previous, std::vector<float> w, std::vector<float> b);
    static std::vector<float> _sigmoid(std::vector<float> l);
    static std::vector<float> _calc_l_next(std::vector<float> l_previous, std::vector<float> w, std::vector<float> b);
    static std::vector<float> _calc_l_last_errors(std::vector<float> l, std::vector<float> l_input_signals, int answer);
    static std::vector<float> _calc_l_hidden_errors(std::vector<float> l_input_signals, std::vector<float> w, std::vector<float> l_next_errors);
    static std::vector<float> _calc_better_b(std::vector<float> b, std::vector<float> errors, float eta);
    static std::vector<float> _calc_better_w(std::vector<float> w, std::vector<float> errors, std::vector<float> l_previous, float eta);
    static std::vector<float> _gen_random_b(int l_size);
    static std::vector<float> _gen_random_w(int l_left_size, int l_right_size);
    int _l1_size, _l2_size, _l3_size, _l4_size;
    std::vector<float> _b2, _b3, _b4;
    std::vector<float> _w12, _w23, _w34;
};