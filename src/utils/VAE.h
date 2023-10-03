#pragma once

#include <torch/script.h>
#include <torch/torch.h>

struct VarEncoder : torch::nn::Module {
    VarEncoder ();
    torch::Tensor forward(torch::Tensor x, torch::Tensor sobol_tensor);
    void load_weights(torch::Tensor encoder_1_weight, torch::Tensor encoder_2_weight, torch::Tensor encoder_3_weight, torch::Tensor encoder_4_weight, torch::Tensor encoder_1_bias, torch::Tensor encoder_2_bias, torch::Tensor encoder_3_bias, torch::Tensor encoder_4_bias);
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};
    torch::nn::Linear fc4{nullptr};
};

struct Decoder : torch::nn::Module {
    Decoder ();
    torch::Tensor forward(torch::Tensor x, torch::Tensor sobol_tensor);
    void load_weights(torch::Tensor decoder_1_weight, torch::Tensor decoder_2_weight, torch::Tensor decoder_3_weight, torch::Tensor decoder_1_bias, torch::Tensor decoder_2_bias, torch::Tensor decoder_3_bias);
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};
};