#include "NeuralNetwork.hpp"


NeuralNetwork::NeuralNetwork(int l1_size, int l2_size, int l3_size, int l4_size) {
    if (l1_size < 1 or l2_size < 1 or l3_size < 1 or l4_size < 1) throw "Fatal error. Invalid size of layers.";
    _l1_size = l1_size;
    _l2_size = l2_size;
    _l3_size = l3_size;
    _l4_size = l4_size;
    _b2 = NeuralNetwork::_gen_random_b(_l2_size);
    _b3 = NeuralNetwork::_gen_random_b(_l3_size);
    _b4 = NeuralNetwork::_gen_random_b(_l4_size);
    _w12 = NeuralNetwork::_gen_random_w(l1_size, l2_size);
    _w23 = NeuralNetwork::_gen_random_w(l2_size, l3_size);
    _w34 = NeuralNetwork::_gen_random_w(l3_size, l4_size);
}
int NeuralNetwork::predict(const NeuralNetwork& nn, std::vector<float> l1) {
    std::vector<float> l2 = NeuralNetwork::_calc_l_next(std::move(l1), nn._w12, nn._b2);
    std::vector<float> l3 = NeuralNetwork::_calc_l_next(l2, nn._w23, nn._b3);
    std::vector<float> l4 = NeuralNetwork::_calc_l_next(l3, nn._w34, nn._b4);
    float max = -1.175494351e38;
    int id = 0;
    for (int i = 0; i < nn._l4_size; i = i + 1) {
        if (l4[i] > max) {
            max = l4[i];
            id = i;
        }
    }
    return id;
}
NeuralNetwork NeuralNetwork::train(NeuralNetwork nn, std::vector<std::vector<float>> data, std::vector<int> answers, float eta, float necessary_MSE, bool log) {
    if (data.size() != answers.size()) throw "Fatal error. Invalid layers.";
    for (int i = 0; i < data.size(); i = i + 1) if (data[i].size() != nn._l1_size) throw "Fatal error. Invalid layers.";
    if (necessary_MSE < 0) throw "Fatal error. Wrong necessary MSE.";
    if (eta <= 0 or eta > 1) throw "Fatal error. Invalid learning rate.";
    std::vector<float>                   l1;
    std::vector<float> l2_input_signals, l2;
    std::vector<float> l3_input_signals, l3;
    std::vector<float> l4_input_signals, l4;
    std::vector<float> l4_errors, l3_errors, l2_errors;
    float errors_sum = 0;
    float MSE;
    int buff;
    int answer;
    std::random_device random_device;
    std::mt19937 mersenne(random_device());
    for (int i = 0; true; i = i + 1) {
        if (log) errors_sum = 0;
        for (int j = 0; j < data.size(); j = j + 1) {
            buff = (int)(mersenne() % answers.size());
            l1 = data[buff];
            answer = answers[buff];

            l2_input_signals = NeuralNetwork::_calc_input_signals(l1, nn._w12, nn._b2);
            l2 = NeuralNetwork::_sigmoid(l2_input_signals);

            l3_input_signals = NeuralNetwork::_calc_input_signals(l2, nn._w23, nn._b3);
            l3 = NeuralNetwork::_sigmoid(l3_input_signals);

            l4_input_signals = NeuralNetwork::_calc_input_signals(l3, nn._w34, nn._b4);
            l4 = NeuralNetwork::_sigmoid(l4_input_signals);

            l4_errors = NeuralNetwork::_calc_l_last_errors(l4, l4_input_signals, answer);
            l3_errors = NeuralNetwork::_calc_l_hidden_errors(l3_input_signals, nn._w34, l4_errors);
            l2_errors = NeuralNetwork::_calc_l_hidden_errors(l2_input_signals, nn._w23, l3_errors);

            nn._b4 = NeuralNetwork::_calc_better_b(nn._b4, l4_errors, eta);
            nn._b3 = NeuralNetwork::_calc_better_b(nn._b3, l3_errors, eta);
            nn._b2 = NeuralNetwork::_calc_better_b(nn._b2, l2_errors, eta);

            nn._w34 = NeuralNetwork::_calc_better_w(nn._w34, l4_errors, l3, eta);
            nn._w23 = NeuralNetwork::_calc_better_w(nn._w23, l3_errors, l2, eta);
            nn._w12 = NeuralNetwork::_calc_better_w(nn._w12, l2_errors, l1, eta);

            if (log) for (int k = 0; k < l4.size(); k = k + 1) {
                    if (answer == k) errors_sum = errors_sum + (float)std::pow((1 - l4[k]), 2);
                    else errors_sum = errors_sum + (float)std::pow(l4[k], 2);
                }
        }
        MSE = errors_sum / (float)(data.size() * nn._l4_size);
        if (log) std::cout << "Epoch = " << i + 1 << ". Necessary MSE = " << necessary_MSE << ". MSE = " << MSE << "." << std::endl;
        if (MSE <= necessary_MSE) return nn;
    }
}
void NeuralNetwork::save(NeuralNetwork nn, const std::string& location) {
    std::ofstream file(location);
    if (!file.is_open()) throw "Fatal error. Invalid file location.";
    file << std::to_string(nn._l1_size) << '\n';
    file << std::to_string(nn._l2_size) << '\n';
    file << std::to_string(nn._l3_size) << '\n';
    file << std::to_string(nn._l4_size) << '\n';
    for (int i = 0; i < nn._b2.size(); i = i + 1) file << std::to_string(nn._b2[i]) << '\n';
    for (int i = 0; i < nn._b3.size(); i = i + 1) file << std::to_string(nn._b3[i]) << '\n';
    for (int i = 0; i < nn._b4.size(); i = i + 1) file << std::to_string(nn._b4[i]) << '\n';
    for (int i = 0; i < nn._w12.size(); i = i + 1) file << std::to_string(nn._w12[i]) << '\n';
    for (int i = 0; i < nn._w23.size(); i = i + 1) file << std::to_string(nn._w23[i]) << '\n';
    for (int i = 0; i < nn._w34.size(); i = i + 1) {
        if (i == nn._w34.size() - 1) file << std::to_string(nn._w34[i]);
        else file << std::to_string(nn._w34[i]) << '\n';
    }
    file.close();
}
NeuralNetwork::NeuralNetwork(const char* location) {
    std::ifstream file(location);
    if (!file.is_open()) throw "Fatal error. Invalid file location.";

    std::string buff;

    std::getline(file, buff); _l1_size = std::stoi(buff);
    std::getline(file, buff); _l2_size = std::stoi(buff);
    std::getline(file, buff); _l3_size = std::stoi(buff);
    std::getline(file, buff); _l4_size = std::stoi(buff);

    _b2.resize(_l2_size);
    _b3.resize(_l3_size);
    _b4.resize(_l4_size);
    for (int i = 0; i < _l2_size; i = i + 1) {std::getline(file, buff); _b2[i] = std::stof(buff);}
    for (int i = 0; i < _l3_size; i = i + 1) {std::getline(file, buff); _b3[i] = std::stof(buff);}
    for (int i = 0; i < _l4_size; i = i + 1) {std::getline(file, buff); _b4[i] = std::stof(buff);}

    _w12.resize(_l1_size * _l2_size);
    _w23.resize(_l2_size * _l3_size);
    _w34.resize(_l3_size * _l4_size);
    for (int i = 0; i < _w12.size(); i = i + 1) {std::getline(file, buff); _w12[i] = std::stof(buff);}
    for (int i = 0; i < _w23.size(); i = i + 1) {std::getline(file, buff); _w23[i] = std::stof(buff);}
    for (int i = 0; i < _w34.size(); i = i + 1) {std::getline(file, buff); _w34[i] = std::stof(buff);}
    file.close();
}
float NeuralNetwork::_sigmoid(float number) {
    return 1 / (1 + std::exp(-number));
}
float NeuralNetwork::_sigmoid_derivative(float number) {
    return NeuralNetwork::_sigmoid(number) * (1 - NeuralNetwork::_sigmoid(number));
}
std::vector<float> NeuralNetwork::_calc_input_signals(std::vector<float> l_previous, std::vector<float> w, std::vector<float> b) {
    std::vector<float> l(b.size());
    for (int i = 0; i < l_previous.size(); i = i + 1) {
        for (int j = 0; j < l.size(); j = j + 1) l[j] = l[j] + l_previous[i] * w[i * l.size() + j];
    }
    for (int i = 0; i < l.size(); i = i + 1) l[i] = l[i] + b[i];
    return l;
}
std::vector<float> NeuralNetwork::_sigmoid(std::vector<float> l) {
    for (int i = 0; i < l.size(); i = i + 1) l[i] = NeuralNetwork::_sigmoid(l[i]);
    return l;
}
std::vector<float> NeuralNetwork::_calc_l_next(std::vector<float> l_previous, std::vector<float> w, std::vector<float> b) {
    std::vector<float> l = NeuralNetwork::_calc_input_signals(std::move(l_previous), std::move(w), std::move(b));
    l = NeuralNetwork::_sigmoid(l);
    return l;
}
std::vector<float> NeuralNetwork::_calc_l_last_errors(std::vector<float> l, std::vector<float> l_input_signals, int answer) {
    std::vector<float> errors(l.size());
    for (int i = 0; i < errors.size(); i = i + 1) {
        if (answer == i) errors[i] = (1 - l[i]) * NeuralNetwork::_sigmoid_derivative(l_input_signals[i]);
        else errors[i] = -l[i] * NeuralNetwork::_sigmoid_derivative(l_input_signals[i]);
    }
    return errors;
}
std::vector<float> NeuralNetwork::_calc_l_hidden_errors(std::vector<float> l_input_signals, std::vector<float> w, std::vector<float> l_next_errors) {
    float sum;
    std::vector<float> errors(l_input_signals.size());
    for (int i = 0; i < errors.size(); i = i + 1) {
        sum = 0;
        for (int j = 0; j < l_next_errors.size(); j = j + 1) sum = sum + l_next_errors[j] * w[i * l_next_errors.size() + j];
        errors[i] = sum * NeuralNetwork::_sigmoid_derivative(l_input_signals[i]);
    }
    return errors;
}
std::vector<float> NeuralNetwork::_calc_better_b(std::vector<float> b, std::vector<float> errors, float eta) {
    for (int i = 0; i < b.size(); i = i + 1) b[i] = b[i] + eta * errors[i];
    return b;
}
std::vector<float> NeuralNetwork::_calc_better_w(std::vector<float> w, std::vector<float> errors, std::vector<float> l_previous, float eta) {
    for (int i = 0; i < l_previous.size(); i = i + 1) {
        for (int j = 0; j < errors.size(); j = j + 1) w[i * errors.size() + j] = w[i * errors.size() + j] + eta * errors[j] * l_previous[i];
    }
    return w;
}
std::vector<float> NeuralNetwork::_gen_random_b(int l_size) {
    std::random_device random_device;
    std::mt19937 mersenne(random_device());
    std::vector<float> b(l_size);
    for (int i = 0; i < l_size; i = i + 1) b[i] = (float)(mersenne() % 1000000) / -2000000;
    return b;
}
std::vector<float> NeuralNetwork::_gen_random_w(int l_left_size, int l_right_size) {
    std::random_device random_device;
    std::mt19937 mersenne(random_device());
    std::vector<float> w(l_left_size * l_right_size);
    for (int i = 0; i < w.size(); i = i + 1) {
        if (mersenne() % 2 == 1) w[i] = (float)(mersenne() % 1000000) / 2000000;
        else w[i] = (float)(mersenne() % 1000000) / -2000000;
    }
    return w;
}