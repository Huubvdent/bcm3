#pragma once

#include <torch/script.h>
#include <torch/torch.h>

struct VarEncoder : torch::nn::Module {
    VarEncoder ();
    at::Tensor forward(at::Tensor x, at::Tensor sobol_tensor);
    void load_weights(at::Tensor encoder_1_weight, at::Tensor encoder_2_weight, at::Tensor encoder_3_weight, at::Tensor encoder_4_weight, at::Tensor encoder_1_bias, at::Tensor encoder_2_bias, at::Tensor encoder_3_bias, at::Tensor encoder_4_bias);
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};
    torch::nn::Linear fc4{nullptr};
};

struct Decoder : torch::nn::Module {
    Decoder ();
    torch::Tensor forward(at::Tensor x, at::Tensor sobol_tensor);
    void load_weights(at::Tensor decoder_1_weight, at::Tensor decoder_2_weight, at::Tensor decoder_3_weight, at::Tensor decoder_1_bias, at::Tensor decoder_2_bias, at::Tensor decoder_3_bias);
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};
};