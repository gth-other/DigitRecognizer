#include <iostream>
#include <array>
#include <vector>
#include <tuple>
#include <Magick++.h>
#include "../src/NeuralNetwork.hpp"


std::array<int, 10> dir_size_in_data_images_training = {5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949};
std::array<int, 10> dir_size_in_data_images_testing = {980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009};
std::vector<float> load_jpg(const std::string& file_location) {
    Magick::Image image;
    image.read( file_location);
    image.type(Magick::TrueColorType);
    image.colorspaceType(Magick::sRGBColorspace);
    size_t width = image.baseColumns();
    size_t height = image.baseRows();
    std::vector<float> vector(width * height);
    Magick::ColorRGB colour;
    for (size_t row = 0; row < height; row = row + 1) {
        for (size_t column = 0; column < width; column = column + 1) {
            colour = image.pixelColor(row, column);
            vector[row * width + column] = (float)(0.2126 * colour.red() + 0.7152 * colour.green() + 0.0722 * colour.blue());
        }
    }
    return vector;
}
std::string set_number_length(int number, int length) {
    std::string result = std::to_string(number);
    while (result.size() < length) result.insert(result.begin(), '0');
    return result;
}
std::tuple<std::vector<std::vector<float>>, std::vector<int>> load_training_data() {
    std::vector<std::vector<float>> data(60000);
    std::vector<int> answers(60000);
    int counter = 0;
    for (int i = 0; i < 10; i = i + 1) {
        for (int j = 0; j < dir_size_in_data_images_training[i]; j = j + 1) {
            data[counter] = load_jpg("../data/images/training/" + std::to_string(i) + "/" + set_number_length(j, 9) + ".jpg");
            answers[counter] = i;
            counter = counter + 1;
            if (counter % 1000 == 0) std::cout << "Загружено " << counter << " примеров из 60000." << std::endl;
        }
    }
    return std::make_tuple(data, answers);
}
std::tuple<std::vector<std::vector<float>>, std::vector<int>> load_testing_data() {
    std::vector<std::vector<float>> data(10000);
    std::vector<int> answers(10000);
    int counter = 0;
    for (int i = 0; i < 10; i = i + 1) {
        for (int j = 0; j < dir_size_in_data_images_testing[i]; j = j + 1) {
            data[counter] = load_jpg("../data/images/testing/" + std::to_string(i) + "/" + set_number_length(j, 9) + ".jpg");
            answers[counter] = i;
            counter = counter + 1;
            if (counter % 1000 == 0) std::cout << "Загружено " << counter << " примеров из 10000." << std::endl;
        }
    }
    return std::make_tuple(data, answers);
}
int main() {
    std::cout << "Доброго времени суток. Перед Вами нейронная сеть, распознающая рукописные цифры." << std::endl;
    std::cout << "1| Загрузить конфигурацию обученной нейронной сети и протестировать ее на заготовленных данных." << std::endl;
    std::cout << "2| Загрузить конфигурацию обученной нейронной сети и протестировать ее на собственном изображении." << std::endl;
    std::cout << "3| Самостоятельно обучить нейронную сеть на заготовленных данных и собственных гиперпараметрах." << std::endl;
    std::cout << "4| Выход." << std::endl;
    std::cout << std::endl;

    std::string buff1;
    std::tuple<std::vector<std::vector<float>>, std::vector<int>> buff2;
    std::vector<std::vector<float>> data;
    std::vector<int> answers;
    int correct = 0;
    int result;
    int l1_size = 784;
    int l2_size;
    int l3_size;
    int l4_size = 10;
    float eta;
    float necessary_MSE;

    std::cout << "Выберите действие: ";
    std::getline(std::cin, buff1);
    std::cout << std::endl;

    if (buff1 == "1") {
        std::cout << "Загрузка нейронной сети." << std::endl;
        NeuralNetwork digit_recognizer = "../data/DigitRecognizer.cfg";
        std::cout << "Загрузка данных для тестирования нейронной сети." << std::endl;
        buff2 = load_testing_data();
        data = std::get<0>(buff2);
        answers = std::get<1>(buff2);
        std::cout << std::endl;

        std::cout << "Оценка качества нейронной сети." << std::endl;
        for (int i = 0; i < data.size(); i = i + 1) if (NeuralNetwork::predict(digit_recognizer, data[i]) == answers[i]) correct = correct + 1;
        std::cout << std::endl;

        std::cout << "Результат: " << (float)correct / (float)data.size() * 100 << "% (" << correct << "/" << data.size() << ")." << std::endl;
    }
    else if (buff1 == "2") {
        std::cout << "Загрузка нейронной сети." << std::endl;
        NeuralNetwork digit_recognizer = "../data/DigitRecognizer.cfg";
        std::cout << std::endl;

        std::cout << "Существует некоторых требования к своим изображениям." << std::endl;
        std::cout << "Изображение должно быть JPG формата в разрешении 28x28." << std::endl;
        std::cout << "Цифра должна быть написана светлым цветом на темном фоне." << std::endl;
        std::cout << "Не забывайте, что цифра и число - разные понятия." << std::endl;
        std::cout << std::endl;

        std::cout << "Укажите полный путь к изображению: ";
        std::getline(std::cin, buff1);
        std::cout << std::endl;

        std::cout << "Выполняется загрузка изображения и расчет." << std::endl;
        result = NeuralNetwork::predict(digit_recognizer, load_jpg(buff1));
        std::cout << std::endl;

        std::cout << "Результат: Вероятно " << result << "." << std::endl;
    }
    else if (buff1 == "3") {
        std::cout << "Загрузка данных для обучения нейронной сети." << std::endl;
        buff2 = load_training_data();
        data = std::get<0>(buff2);
        answers = std::get<1>(buff2);
        std::cout << std::endl;

        std::cout << "l1_size = " << l1_size << std::endl;
        std::cout << "l2_size = "; std::getline(std::cin, buff1); l2_size = std::stoi(buff1);
        std::cout << "l3_size = "; std::getline(std::cin, buff1); l3_size = std::stoi(buff1);
        std::cout << "l4_size = " << l4_size << std::endl;
        std::cout << "eta = "; std::getline(std::cin, buff1); eta = std::stof(buff1);
        std::cout << "necessary_MSE = "; std::getline(std::cin, buff1); necessary_MSE = std::stof(buff1);
        std::cout << std::endl;

        std::cout << "Инициализация и обучение нейронной сети." << std::endl;
        NeuralNetwork digit_recognizer = {l1_size, l2_size, l3_size, l4_size};
        digit_recognizer = NeuralNetwork::train(digit_recognizer, data, answers, eta, necessary_MSE, true);
        std::cout << std::endl;

        std::cout << "Загрузка данных для тестирования нейронной сети." << std::endl;
        buff2 = load_testing_data();
        data = std::get<0>(buff2);
        answers = std::get<1>(buff2);
        std::cout << std::endl;

        std::cout << "Оценка качества нейронной сети." << std::endl;
        for (int i = 0; i < data.size(); i = i + 1) if (NeuralNetwork::predict(digit_recognizer, data[i]) == answers[i]) correct = correct + 1;
        std::cout << std::endl;

        std::cout << "Результат: " << (float)correct / (float)data.size() * 100 << "% (" << correct << "/" << data.size() << ")." << std::endl;
        std::cout << std::endl;

        std::cout << "Вы хотите сохранить обученную нейронную сеть (1 - Да, 2 - Нет): ";
        std::getline(std::cin, buff1);
        if (buff1 == "1") {NeuralNetwork::save(digit_recognizer, "../data/DigitRecognizer.cfg"); std::cout << "Сохранено." << std::endl;}
        else if (buff1 == "2") std::cout << "Процесс завершен." << std::endl;
        else std::cout << "Неизвестное действие." << std::endl;
    }
    else if (buff1 == "4") std::cout << "Процесс завершен." << std::endl;
    else std::cout << "Неизвестное действие." << std::endl;
    return 0;
}