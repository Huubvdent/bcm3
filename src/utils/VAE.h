#pragma once

#include <torch/script.h>
#include <torch/torch.h>

namespace bcm3{

struct VarEncoder : torch::nn::Module {
    VarEncoder ();
    at::Tensor forward(at::Tensor x, at::Tensor sobol_tensor);
    void load_weights(at::Tensor encoder_1_weight, at::Tensor encoder_2_weight, at::Tensor encoder_3_weight, at::Tensor encoder_1_bias, at::Tensor encoder_2_bias, at::Tensor encoder_3_bias);
    at::Tensor get_weights_fc1();
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};
};

struct Decoder : torch::nn::Module {
    Decoder ();
    at::Tensor forward(at::Tensor x);
    void load_weights(at::Tensor decoder_1_weight, at::Tensor decoder_2_weight, at::Tensor decoder_1_bias, at::Tensor decoder_2_bias);
    at::Tensor get_weights_fc1();
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
};

}