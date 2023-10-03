#pragma once

#include <torch/script.h>
#include <torch/torch.h>

struct VarEncoder : torch::jit::nn::Module {
    VarEncoder () {};
    torch::Tensor forward(torch::Tensor x, torch::Tensor sobol_tensor);
    torch::jit::nn::Linear fc1;
    torch::jit::nn::Linear fc2;
    torch::jit::nn::Linear fc3;
    torch::jit::nn::Linear fc4;
};

struct Decoder : torch::jit::nn::Module {
    Decoder () {};
    torch::Tensor forward(torch::Tensor x, torch::Tensor sobol_tensor) {};
    torch::jit::nn::Linear fc1;
    torch::jit::nn::Linear fc2;
    torch::jit::nn::Linear fc3;
};