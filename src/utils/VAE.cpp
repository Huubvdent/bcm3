#include "VAE.h"

#include <torch/script.h>
#include <torch/torch.h>

struct VarEncoder : torch::nn::Module 
{
    VarEncoder () 
    {
        fc1 = register_module("fc1", torch::nn::Linear(13, 10));
        fc2 = register_module("fc2", torch::nn::Linear(10, 6));
        fc3 = register_module("fc3", torch::nn::Linear(6, 2));
        fc4 = register_module("fc4", torch::nn::Linear(6, 2));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor sobol_tensor) 
    {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        mu = torch::relu(fc3->forward(x));
        sigma = torch::relu(fc4->forward(x));
        // Use sobol sequence to sample deterministically
        z = mu + sigma * sobol_tensor;
        return z;
    }

    void load_weights(torch::Tensor encoder_1_weight, torch::Tensor encoder_2_weight, torch::Tensor encoder_3_weight, torch::Tensor encoder_4_weight, torch::Tensor encoder_1_bias, torch::Tensor encoder_2_bias, torch::Tensor encoder_3_bias, torch::Tensor encoder_4_bias)
    {
        fc1->weight = encoder_1_weight;
        fc2->weight = encoder_2_weight;
        fc3->weight = encoder_3_weight;
        fc4->weight = encoder_4_weight;

        fc1->bias = encoder_1_bias;
        fc2->bias = encoder_2_bias;
        fc3->bias = encoder_3_bias;
        fc4->bias = encoder_4_bias;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};

};

struct Decoder : torch::nn::Module {
    Decoder () 
    {
        fc1 = register_module("fc1", torch::nn::Linear(2, 6));
        fc2 = register_module("fc2", torch::nn::Linear(6, 10));
        fc3 = register_module("fc3", torch::nn::Linear(10, 13));
    }

    torch::Tensor forward(torch::Tensor x) 
    {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = torch::sigmoid(fc3->forward(x));
        return x;
    }

    void load_weights(torch::Tensor decoder_1_weight, torch::Tensor decoder_2_weight, torch::Tensor decoder_3_weight, torch::Tensor decoder_1_bias, torch::Tensor decoder_2_bias, torch::Tensor decoder_3_bias)
    {
        fc1->weight = decoder_1_weight;
        fc2->weight = decoder_2_weight;
        fc3->weight = decoder_3_weight;

        fc1->bias = decoder_1_bias;
        fc2->bias = decoder_2_bias;
        fc3->bias = decoder_3_bias;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

};