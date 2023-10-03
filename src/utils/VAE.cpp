#include "VAE.h"

#include <torch/script.h>

struct VarEncoder : torch::jit::nn::Module {
    VarEncoder () {
        fc1 = register_module("fc1", torch::nn::Linear(13, 10));
        fc2 = register_module("fc2", torch::nn::Linear(10, 6));
        fc3 = register_module("fc3", torch::nn::Linear(6, 2));
        fc4 = register_module("fc4", torch::nn::Linear(6, 2));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor sobol_tensor) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        mu = torch::relu(fc3->forward(x));
        sigma = torch::relu(fc4->forward(x));
        // Use sobol sequence to sample deterministically
        z = mu + sigma * sobol_tensor;
        return z;
    }

    torch::jit::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};

};

struct Decoder : torch::jit::nn::Module {
    Decoder () {
        fc1 = register_module("fc1", torch::nn::Linear(2, 6));
        fc2 = register_module("fc2", torch::nn::Linear(6, 10));
        fc3 = register_module("fc3", torch::nn::Linear(10, 13));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = torch::sigmoid(fc3->forward(x));
        return x;
    }

    torch::jit::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

};