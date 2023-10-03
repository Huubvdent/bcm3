#pragma once

#include <torch/script.h>

struct VarEncoder : torch::nn::Module {
    VarEncoder () {};
    torch::Tensor forward(torch::Tensor x, torch::Tensor sobol_tensor);
};

struct Decoder : torch::nn::Module {
    Decoder () {};
    torch::Tensor forward(torch::Tensor x, torch::Tensor sobol_tensor) {};
};