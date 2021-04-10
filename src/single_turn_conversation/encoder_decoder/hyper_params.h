#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_HYPER_PARAMS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_HYPER_PARAMS_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <boost/format.hpp>
#include <string>
#include "n3ldg-plus/n3ldg-plus.h"

using std::string;
using namespace ::n3ldg_plus;

enum Optimizer {
    ADAM = 0,
    ADAGRAD = 1,
    ADAMW = 2
};

struct HyperParams {
    int hidden_dim;
    int hidden_layer;
    int head_count;
    float dropout;
    int batch_size;
    int beam_size;
    float learning_rate;
    bool lr_decay;
    int warm_up_iterations;
    int word_cutoff;
    float l2_reg;
    ::Optimizer optimizer;
    string word_file;

    template<typename Archive>
    void serialize(Archive &ar) {
        ar(hidden_dim, hidden_layer, head_count, dropout, batch_size, beam_size, learning_rate,
                lr_decay, warm_up_iterations, word_cutoff, l2_reg, optimizer);
    }

    void print() const {
        std::cout << "hidden_dim:" << hidden_dim << std::endl
            << "hidden_layer:" << hidden_layer << std::endl
            << "head_count:" << head_count << std::endl
            << "dropout:" << dropout << std::endl
            << "batch_size:" << batch_size << std::endl
            << "beam_size:" << beam_size << std::endl
            << "learning_rate:" << learning_rate << std::endl
            << "lr_decay:" << lr_decay << std::endl
            << "warm_up_iterations:" << warm_up_iterations << std::endl
	    << "word_cutoff:" << word_cutoff << std::endl
    	    << "l2_reg:" << l2_reg << std::endl
    	    << "word_file:" << word_file << std::endl
    	    << "optimizer:" << optimizer << std::endl; 
    }
};

#endif
