#include "Perceptron.hpp"


Perceptron::Perceptron() = default;
Perceptron::Perceptron(const std::vector<int32_t> &layersSize) {
    this->layersSize = layersSize;

    this->weights.resize(this->layersSize.size() - 1);
    for (int32_t i = 0; i < this->weights.size(); i = i + 1) {
        this->weights[i] = Perceptron::randomWeights(this->layersSize[i], this->layersSize[i + 1]);
    }

    this->biases.resize(this->layersSize.size() - 1);
    for (int32_t i = 0; i < this->biases.size(); i = i + 1) {
        this->biases[i] = Perceptron::randomBiases(this->layersSize[i + 1]);
    }
}
Perceptron::Perceptron(const std::string &path) {
    int32_t buff;

    std::ifstream file(path);

    buff = Perceptron::readFromBinaryFile(file);
    this->layersSize.resize(buff);
    this->weights.resize(buff - 1);
    this->biases.resize(buff- 1);

    for (int32_t i = 0; i < this->layersSize.size(); i = i + 1) {
        buff = Perceptron::readFromBinaryFile(file);
        this->layersSize[i] = buff;
    }
    for (int32_t i = 0; i < this->weights.size(); i = i + 1) {
        this->weights[i] = Tensor2D(this->layersSize[i], this->layersSize[i + 1]);
        for (int32_t j = 0; j < this->layersSize[i] * this->layersSize[i + 1]; j = j + 1) {
            buff = Perceptron::readFromBinaryFile(file);
            this->weights[i](j) = (float)(buff / 1e+6);
        }
    }
    for (int32_t i = 0; i < this->biases.size(); i = i + 1) {
        this->biases[i].resize(this->layersSize[i + 1]);
        for (int32_t j = 0; j < this->layersSize[i + 1]; j = j + 1) {
            buff = Perceptron::readFromBinaryFile(file);
            this->biases[i][j] = (float)(buff / 1e+6);
        }
    }

    file.close();
}
std::vector<float> Perceptron::eval(std::vector<float> l1) const {
    for (int32_t i = 0; i < this->weights.size(); i = i + 1) {
        l1 = Perceptron::next(l1, this->weights[i], this->biases[i]);
    }

    return l1;
}
void Perceptron::train(const std::vector<std::pair<std::vector<float>, std::vector<float>>> &data, float eta, int32_t necessarySuccessRate, int32_t epochsWithoutImprovement) {
    std::random_device random_device;
    std::mt19937 mersenne(random_device());

    std::vector<std::vector<float>> inputSignals(this->layersSize.size() - 1);
    std::vector<std::vector<float>> layers(this->layersSize.size());
    std::vector<float> perfectAnswer;

    int32_t previousSuccessRate = -1;
    int32_t epochsWithoutImprovementPassed = 1;

    for (int32_t epoch = 0; true; epoch = epoch + 1) {
        int32_t correctOutputs = 0;
        for (int32_t example = 0; example < data.size(); example = example + 1) {
            uint32_t buff = mersenne() % data.size();
            layers[0] = data[buff].first;
            perfectAnswer = data[buff].second;

            for (int32_t k = 1; k < layers.size(); k = k + 1) {
                inputSignals[k - 1] = Perceptron::inputSignals(layers[k - 1], this->weights[k - 1], this->biases[k - 1]);
                layers[k] = Perceptron::sigmoid(inputSignals[k - 1]);
            }

            std::vector<float> error = Perceptron::lastLayerError(layers.back(), inputSignals.back(), perfectAnswer);
            this->weights.back() = Perceptron::updateWeights(this->weights.back(), error, layers[layers.size() - 2], eta);
            this->biases.back() = Perceptron::updateBiases(this->biases.back(), error, eta);

            for (uint32_t k = layers.size() - 2; k > 0; k = k - 1) {
                error = Perceptron::hiddenLayerError(inputSignals[k - 1], this->weights[k], error);
                this->weights[k - 1] = Perceptron::updateWeights(this->weights[k - 1], error, layers[k - 1], eta);
                this->biases[k] = Perceptron::updateBiases(this->biases[k], error, eta);
            }

            int64_t maxIndexGotten = std::distance(layers.back().begin(), std::max_element(layers.back().begin(), layers.back().end()));
            int64_t maxIndexPerfect = std::distance(perfectAnswer.begin(), std::max_element(perfectAnswer.begin(), perfectAnswer.end()));
            correctOutputs = correctOutputs + (maxIndexGotten == maxIndexPerfect);

            if ((example + 1) % (data.size() / 100) == 0) {
                std::cout << "\33[2K\r";
                std::cout << "Epoch = " << epoch + 1 << ". Progress: " << (example + 1) / (data.size() / 100) << "%.";
                std::cout.flush();
            }
        }
        int32_t successRate = correctOutputs * 100 / (int32_t)data.size();
        if (successRate == previousSuccessRate) {
            epochsWithoutImprovementPassed = epochsWithoutImprovementPassed + 1;
        }
        else {
            previousSuccessRate = successRate;
            epochsWithoutImprovementPassed = 1;
        }
        std::cout << "\rEpoch = " << epoch + 1 << ". Necessary success rate = " << necessarySuccessRate << "%. Success rate = " << successRate << "%." << std::endl;
        if (successRate >= necessarySuccessRate) {
            return;
        }
        if (epochsWithoutImprovement == epochsWithoutImprovementPassed) {
            std::cout << "No improvement in " << epochsWithoutImprovement << " epochs." << std::endl;
            return;
        }
    }
}
void Perceptron::save(const std::string &path) {
    std::ofstream file(path);
    Perceptron::writeToBinaryFile(file, (int32_t)this->layersSize.size());
    for (int32_t i = 0; i < this->layersSize.size(); i = i + 1) {
        Perceptron::writeToBinaryFile(file, this->layersSize[i]);
    }
    for (int32_t i = 0; i < this->weights.size(); i = i + 1) {
        for (int32_t j = 0; j < this->weights[i].getSize(); j = j + 1) {
            Perceptron::writeToBinaryFile(file, (int32_t)(this->weights[i](j) * 1e+6));
        }
    }
    for (int32_t i = 0; i < this->biases.size(); i = i + 1) {
        for (int32_t j = 0; j < this->biases[i].size(); j = j + 1) {
            Perceptron::writeToBinaryFile(file, (int32_t)(this->biases[i][j] * 1e+6));
        }
    }
    file.close();
}
void Perceptron::show() const {
    std::cout << Perceptron::Separator;
    int32_t neurons = 0;
    for (int32_t i = 0; i < this->layersSize.size(); i = i + 1) {
        std::cout << "Layer " << i + 1 << ": " << this->layersSize[i] << " neurons\n";
        neurons = neurons + this->layersSize[i];
    }
    std::cout << Perceptron::Separator;

    int32_t parameters = 0;
    for (int32_t i = 0; i < this->weights.size(); i = i + 1) {
        std::cout << "Weights " << i + 1 << "-" << i + 2 << ": " << this->weights[i].getSize() << " parameters\n";
        parameters = parameters + this->weights[i].getSize();
    }
    std::cout << Perceptron::Separator;

    for (int32_t i = 0; i < this->biases.size(); i = i + 1) {
        std::cout << "Biases " << i + 2 << ": " << this->biases[i].size() << " parameters\n";
        parameters = parameters + (int32_t)this->biases[i].size();
    }
    std::cout << Perceptron::Separator;

    std::cout << "Total neurons: " << neurons << '\n';
    std::cout << "Total parameters: " << parameters << " [" << (float)parameters * sizeof(float) / 1024.f / 1024.f << " MB]\n";
    std::cout << Perceptron::Separator;
    std::cout.flush();
}
float Perceptron::sigmoid(float number) {
    return 1 / (1 + std::exp(-number));
}
float Perceptron::sigmoidDerivative(float number) {
    return Perceptron::sigmoid(number) * (1 - Perceptron::sigmoid(number));
}
std::vector<float> Perceptron::inputSignals(const std::vector<float> &layer, const Tensor2D &weights, std::vector<float> biases) {
    for (int32_t i = 0; i < layer.size(); i = i + 1) {
        for (int32_t j = 0; j < biases.size(); j = j + 1) {
            biases[j] = biases[j] + layer[i] * weights(i, j);
        }
    }

    return biases;
}
std::vector<float> Perceptron::sigmoid(std::vector<float> layer) {
    for (int32_t i = 0; i < layer.size(); i = i + 1) {
        layer[i] = Perceptron::sigmoid(layer[i]);
    }

    return layer;
}
std::vector<float> Perceptron::next(const std::vector<float>& layer, const Tensor2D &weights, const std::vector<float> &biases) {
    return Perceptron::sigmoid(Perceptron::inputSignals(layer, weights, biases));
}
std::vector<float> Perceptron::lastLayerError(const std::vector<float> &layer, const std::vector<float> &inputSignals, const std::vector<float> &perfectAnswer) {
    std::vector<float> error(layer.size());

    for (int32_t i = 0; i < error.size(); i = i + 1) {
        error[i] = (perfectAnswer[i] - layer[i]) * Perceptron::sigmoidDerivative(inputSignals[i]);
    }

    return error;
}
std::vector<float> Perceptron::hiddenLayerError(const std::vector<float> &inputSignals, const Tensor2D &weights, const std::vector<float> &error) {
    std::vector<float> backpropError(inputSignals.size());

    for (int32_t i = 0; i < backpropError.size(); i = i + 1) {
        float sum = 0;
        for (int32_t j = 0; j < error.size(); j = j + 1) {
            sum = sum + error[j] * weights(i, j);
        }
        backpropError[i] = sum * Perceptron::sigmoidDerivative(inputSignals[i]);
    }

    return backpropError;
}
Tensor2D Perceptron::updateWeights(Tensor2D weights, const std::vector<float> &error, const std::vector<float> &layer, float eta) {
    for (int32_t i = 0; i < layer.size(); i = i + 1) {
        for (int32_t j = 0; j < error.size(); j = j + 1) {
            weights(i, j) = weights(i, j) + eta * error[j] * layer[i];
        }
    }

    return weights;
}
std::vector<float> Perceptron::updateBiases(std::vector<float> biases, const std::vector<float> &error, float eta) {
    for (int32_t i = 0; i < biases.size(); i = i + 1) {
        biases[i] = biases[i] + eta * error[i];
    }

    return biases;
}
Tensor2D Perceptron::randomWeights(int32_t layerLeftSize, int32_t layerRightSize) {
    std::random_device random_device;
    std::mt19937 mersenne(random_device());

    Tensor2D weights = {layerLeftSize, layerRightSize};
    for (int32_t i = 0; i < weights.getSize(); i = i + 1) {
        weights(i) = (float)mersenne() / (float)std::numeric_limits<uint32_t>::max();
        if (mersenne() % 2) {
            weights(i) = -weights(i);
        }
    }

    return weights;
}
std::vector<float> Perceptron::randomBiases(int32_t layerSize) {
    std::random_device random_device;
    std::mt19937 mersenne(random_device());

    std::vector<float> biases(layerSize);
    for (int32_t i = 0; i < layerSize; i = i + 1) {
        biases[i] = (float)mersenne() / (float)std::numeric_limits<uint32_t>::max();
        if (mersenne() % 2) {
            biases[i] = -biases[i];
        }
    }

    return biases;
}
void Perceptron::writeToBinaryFile(std::ofstream &ofstream, int32_t number) {
    ofstream.write(reinterpret_cast<const char *>(&number), sizeof(number));
}
int32_t Perceptron::readFromBinaryFile(std::ifstream &ifstream) {
    int32_t result;
    ifstream.read((char *)&result, sizeof(result));

    return result;
}