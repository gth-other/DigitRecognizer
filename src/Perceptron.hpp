#include <iostream>
#include <random>
#include <cmath>
#include <limits>
#include <algorithm>
#include <fstream>
#include "Tensor2D.hpp"


#pragma once


class Perceptron {
public:
    Perceptron();
    explicit Perceptron(const std::vector<int32_t> &layersSize);
    explicit Perceptron(const std::string &path);

    [[nodiscard]] std::vector<float> eval(std::vector<float> l1) const;
    void train(const std::vector<std::pair<std::vector<float>, std::vector<float>>> &data, float eta, int32_t necessarySuccessRate, int32_t epochsWithoutImprovement);
    void save(const std::string &path);
    void show() const;
private:
    static float sigmoid(float number);
    static float sigmoidDerivative(float number);

    static std::vector<float> inputSignals(const std::vector<float> &layer, const Tensor2D &weights, std::vector<float> biases);
    static std::vector<float> sigmoid(std::vector<float> layer);
    static std::vector<float> next(const std::vector<float>& layer, const Tensor2D &weights, const std::vector<float> &biases);

    static std::vector<float> lastLayerError(const std::vector<float> &layer, const std::vector<float> &inputSignals, const std::vector<float> &perfectAnswer);
    static std::vector<float> hiddenLayerError(const std::vector<float> &inputSignals, const Tensor2D &weights, const std::vector<float> &error);

    static Tensor2D updateWeights(Tensor2D weights, const std::vector<float> &error, const std::vector<float> &layer, float eta);
    static std::vector<float> updateBiases(std::vector<float> biases, const std::vector<float> &error, float eta);

    static Tensor2D randomWeights(int32_t layerLeftSize, int32_t layerRightSize);
    static std::vector<float> randomBiases(int32_t layerSize);

    static void writeToBinaryFile(std::ofstream &ofstream, int32_t number);
    static int32_t readFromBinaryFile(std::ifstream &ifstream);

    std::vector<int32_t> layersSize;
    std::vector<Tensor2D> weights;
    std::vector<std::vector<float>> biases;

    static constexpr std::string_view Separator = "################################################################################\n";
};