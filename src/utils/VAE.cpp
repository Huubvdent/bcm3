#include "VAE.h"

namespace bcm3{

VarEncoder::VarEncoder () 
      : fc1(register_module("fc1", torch::nn::Linear(13, 8))),
        fc2(register_module("fc2", torch::nn::Linear(8, 2))),
        fc3(register_module("fc3", torch::nn::Linear(8, 2)))
{}

at::Tensor VarEncoder::forward(at::Tensor x, at::Tensor sobol_tensor) 
{
    at::Tensor first = torch::relu(fc1->forward(x));
    at::Tensor mu = torch::relu(fc2->forward(first));
    at::Tensor sigma = torch::exp(fc3->forward(first));
    // Use sobol sequence to sample deterministically
    at::Tensor z = mu + sigma * sobol_tensor;
    return z;
}

void VarEncoder::load_weights(at::Tensor encoder_1_weight, at::Tensor encoder_2_weight, at::Tensor encoder_3_weight, at::Tensor encoder_1_bias, at::Tensor encoder_2_bias, at::Tensor encoder_3_bias)
{
    fc1->weight = encoder_1_weight;
    fc2->weight = encoder_2_weight;
    fc3->weight = encoder_3_weight;

    fc1->bias = encoder_1_bias;
    fc2->bias = encoder_2_bias;
    fc3->bias = encoder_3_bias;
}

at::Tensor VarEncoder::get_weights_fc1(){
    return fc1->weight;
}



Decoder::Decoder () 
        : fc1(register_module("fc1", torch::nn::Linear(2, 8))),
          fc2(register_module("fc2", torch::nn::Linear(8, 13)))
{}

at::Tensor Decoder::forward(at::Tensor x) 
{
    at::Tensor first = torch::relu(fc1->forward(x));
    at::Tensor second = torch::sigmoid(fc2->forward(first));
    return second;
}

void Decoder::load_weights(at::Tensor decoder_1_weight, at::Tensor decoder_2_weight, at::Tensor decoder_1_bias, at::Tensor decoder_2_bias)
{
    fc1->weight = decoder_1_weight;
    fc2->weight = decoder_2_weight;

    fc1->bias = decoder_1_bias;
    fc2->bias = decoder_2_bias;
}

at::Tensor Decoder::get_weights_fc1(){
    return fc1->weight;
}

}