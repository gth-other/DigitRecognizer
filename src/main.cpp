#include <sstream>
#include <SFML/Graphics.hpp>
#include "Perceptron.hpp"


std::vector<float> loadLineFromMNIST(const std::string &string, char separator) {
    std::vector<float> tokens;

    bool first = true;

    float i;
    std::stringstream ss(string);
    while (ss >> i) {
        if (first) {
            tokens.push_back(i);
            first = false;
        }
        else {
            tokens.push_back(i / 255);
        }
        if (ss.peek() == separator) {
            ss.ignore();
        }
    }

    return tokens;
}
std::vector<float> transformMNISTImage(const sf::Image &originalImage) {
    // Очень неэффективный алгоритм, но ничего умнее придумать не получилось.

    std::random_device randomDevice;
    std::mt19937 mersenne(randomDevice());

    sf::Texture texture;
    texture.loadFromImage(originalImage);
    texture.setSmooth(true);

    float scale = (50 + (float)(mersenne() % 150)) / 100;
    float angle = 2 * ((float)(mersenne() % 2) - 0.5f) * (float)(mersenne() % 45);
    float offsetX = 2 * ((float)(mersenne() % 2) - 0.5f) * (float)(mersenne() % 10);
    float offsetY = 2 * ((float)(mersenne() % 2) - 0.5f) * (float)(mersenne() % 10);

    sf::Sprite sprite;
    sprite.setTexture(texture);
    sprite.setPosition(14 + offsetX, 14 + offsetY);
    sprite.setScale(scale, scale);
    sprite.setRotation(angle);

    sf::RenderTexture renderer;
    renderer.create(56, 56);
    renderer.draw(sprite);
    renderer.display();

    sf::Image resultImage = renderer.getTexture().copyToImage();

    for (int32_t y = 0; y < 56; y = y + 1) {
        for (int32_t x = 0; x < 56; x = x + 1) {
            if (((x < 14 or x >= 42) or (y < 14 or y >= 42)) and resultImage.getPixel(x, y).r != 0) {
                return transformMNISTImage(originalImage);
            }
        }
    }

    std::vector<float> resultPixels(784);

    for (int32_t y = 14; y < 42; y = y + 1) {
        for (int32_t x = 14; x < 42; x = x + 1) {
            resultPixels[(y - 14) * 28 + x - 14] = (float)resultImage.getPixel(x, y).r / 255;
        }
    }

    return resultPixels;
}
std::vector<float> transformMNISTImage(const std::vector<float> &originalPixels) {
    sf::Image originalImage;
    originalImage.create(28, 28);
    for (int32_t y = 0; y < 28; y = y + 1) {
        for (int32_t x = 0; x < 28; x = x + 1) {
            auto color = (int32_t)(originalPixels[y * 28 + x] * 255);
            originalImage.setPixel(x, y, sf::Color(color, color, color));
        }
    }

    return transformMNISTImage(originalImage);
}
int main() {
    /*sf::Context settings; // Это нужно из-за бага SFML.

    std::vector<std::pair<std::vector<float>, std::vector<float>>> originalData(60000);

    std::ifstream file;
    std::string buff1;
    std::vector<float> buff2;

    for (int32_t i = 0; i < 6; i = i + 1) { // База примеров разделена, чтобы обойти ограничение гитхаба на максимальный размер файла.
        file.open("../data/mnist" + std::to_string(i + 1) + ".txt");

        for (int32_t j = 0; j < 10000; j = j + 1) {
            std::getline(file, buff1);
            buff2 = loadLineFromMNIST(buff1, ',');

            originalData[10000 * i + j].second.resize(10);
            originalData[10000 * i + j].second[(int32_t)std::round(buff2.front())] = 1;

            buff2.erase(buff2.begin());
            originalData[10000 * i + j].first = buff2;

            if ((10000 * i + j + 1) % 1000 == 0) {
                std::cout << "Загружено " << (10000 * i + j + 1) / 1000 << "k примеров..." << std::endl;
            }
        }

        file.close();
    }

    std::vector<std::pair<std::vector<float>, std::vector<float>>> transformedData(180000);
    for (int32_t i = 0; i < 3; i = i + 1) {
        for (int32_t j = 0; j < 60000; j = j + 1) {
            transformedData[i * 60000 + j] = std::make_pair(transformMNISTImage(originalData[j].first), originalData[j].second);
            if ((i * 60000 + j + 1) % 1000 == 0) {
                std::cout << "Искажено " << (i * 60000 + j + 1) / 1000 << "k примеров..." << std::endl;
            }
        }
    }

    auto perceptron = Perceptron((std::vector<int32_t>){784, 512, 256, 128, 64, 32, 16, 10});
    perceptron.show();
    perceptron.train(transformedData, 0.01, 100, 3);
    perceptron.save("../data/perceptron.txt");*/

    auto perceptron = Perceptron("../data/perceptron.hex");
    perceptron.show();

    auto window = sf::RenderWindow(sf::VideoMode(800, 500), "untitled", sf::Style::Titlebar | sf::Style::Close);
    sf::Event event{};
    window.setFramerateLimit(240);

    sf::RectangleShape border;
    border.setPosition(10, 10);
    border.setSize(sf::Vector2f(280, 280));
    border.setOutlineThickness(2);
    border.setOutlineColor(sf::Color(216, 216, 216));
    border.setFillColor(sf::Color::Transparent);

    sf::Font font;
    font.loadFromFile("../data/bitterProMedium.ttf");

    sf::Text perceptronOutput;
    perceptronOutput.setPosition(border.getPosition().x + border.getSize().x + border.getOutlineThickness() + 10, border.getPosition().y - border.getOutlineThickness());
    perceptronOutput.setFillColor(sf::Color(90, 101, 102));
    perceptronOutput.setCharacterSize(14);
    perceptronOutput.setFont(font);

    sf::Text docs;
    docs.setFillColor(sf::Color(90, 101, 102));
    docs.setCharacterSize(14);
    docs.setFont(font);
    docs.setString(L"Нажмите на пробел, чтобы очистить поле для рисования");
    docs.setPosition(10, (float)window.getSize().y - docs.getLocalBounds().height - 10);

    std::vector<float> picture(784);

    bool mouseButtonHolding = false;

    for (; ;) {
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                return 0;
            }
            else if (event.type == sf::Event::MouseButtonPressed) {
                mouseButtonHolding = true;
            }
            else if (event.type == sf::Event::MouseButtonReleased) {
                mouseButtonHolding = false;
            }
            else if (event.type == sf::Event::KeyPressed and event.key.code == sf::Keyboard::Space) {
                picture = std::vector<float>(784);
            }
        }

        if (mouseButtonHolding) {
            float x = (float)sf::Mouse::getPosition(window).x - border.getPosition().x;
            float y = (float)sf::Mouse::getPosition(window).y - border.getPosition().y;
            if (x >= 0 and y >= 0 and x < border.getSize().x and y < border.getSize().y) {
                auto xi = (int32_t)std::round(x / (border.getSize().x / 28));
                auto yi = (int32_t)std::round(y / (border.getSize().y / 28));
                for (int32_t i = 0; i <= 28; i = i + 28) {
                    for (int32_t j = 0; j <= 1; j = j + 1) {
                        int32_t index = yi * 28 + xi + i + j;
                        if (index >= 0 and index < 784 and (index - i) / 28 == yi) {
                            picture[yi * 28 + xi + i + j] = 1;
                        }
                    }
                }
            }
            std::vector<float> result = perceptron.eval(picture);
            perceptronOutput.setString(sf::String());
            for (int32_t i = 0; i < 10; i = i + 1) {
                float max = -1;
                int32_t maxID;
                for (int32_t j = 0; j < 10; j = j + 1) {
                    if (result[j] > max) {
                        max = result[j];
                        maxID = j;
                    }
                }
                result[maxID] = -1;
                switch (maxID) {
                    case 0: {perceptronOutput.setString(perceptronOutput.getString() + L"Ноль       "); break;}
                    case 1: {perceptronOutput.setString(perceptronOutput.getString() + L"Один       "); break;}
                    case 2: {perceptronOutput.setString(perceptronOutput.getString() + L"Два        "); break;}
                    case 3: {perceptronOutput.setString(perceptronOutput.getString() + L"Три        "); break;}
                    case 4: {perceptronOutput.setString(perceptronOutput.getString() + L"Четыре     "); break;}
                    case 5: {perceptronOutput.setString(perceptronOutput.getString() + L"Пять       "); break;}
                    case 6: {perceptronOutput.setString(perceptronOutput.getString() + L"Шесть      "); break;}
                    case 7: {perceptronOutput.setString(perceptronOutput.getString() + L"Семь       "); break;}
                    case 8: {perceptronOutput.setString(perceptronOutput.getString() + L"Восемь     "); break;}
                    case 9: {perceptronOutput.setString(perceptronOutput.getString() + L"Девять     "); break;}
                }
                perceptronOutput.setString(perceptronOutput.getString() + std::to_string((int32_t)(max * 100)) + "%\n");
            }
        }


        window.clear(sf::Color(250, 250, 250));

        for (int32_t x = 0; x < 28; x = x + 1) {
            for (int32_t y = 0; y < 28; y = y + 1) {
                auto brightness = (int32_t)((1 - picture[y * 28 + x]) * 255);
                sf::RectangleShape buff;
                buff.setFillColor(sf::Color(brightness, brightness, brightness));
                buff.setSize(sf::Vector2f(border.getSize().x / 28, border.getSize().y / 28));
                buff.setPosition(border.getPosition().x + border.getSize().x / 28 * (float)x, border.getPosition().y + border.getSize().y / 28 * (float)y);
                window.draw(buff);
            }
        }

        window.draw(border);
        window.draw(perceptronOutput);
        window.draw(docs);

        window.display();
    }
}