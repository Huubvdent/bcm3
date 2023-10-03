#pragma once

#include <torch/script.h>

struct VarEncoder : torch::nn::Module {
    VarEncoder () {};
    torch::Tensor forward(torch::Tensor x, torch::Tensor sobol_tensor);
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::Linear fc3;
    torch::nn::Linear fc4;
};

struct Decoder : torch::nn::Module {
    Decoder () {};
    torch::Tensor forward(torch::Tensor x, torch::Tensor sobol_tensor) {};
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::Linear fc3;
};