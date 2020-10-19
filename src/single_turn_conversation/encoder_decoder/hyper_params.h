#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_HYPER_PARAMS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_HYPER_PARAMS_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <boost/format.hpp>
#include <string>
#include "serializable.h"

using std::string;

enum Optimizer {
    ADAM = 0,
    ADAGRAD = 1,
    ADAMW = 2
};

struct HyperParams : public N3LDGSerializable {
    int word_dim;
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
    string word_file;
    bool word_finetune;
    float l2_reg;
    Optimizer optimizer;

    Json::Value toJson() const override {
        Json::Value json;
        json["word_dim"] = word_dim;
        json["hidden_dim"] = hidden_dim;
        json["hidden_layer"] = hidden_layer;
        json["head_count"] = head_count;
        json["dropout"] = dropout;
        json["batch_size"] = batch_size;
        json["beam_size"] = beam_size;
        json["learning_rate"] = learning_rate;
        json["lr_decay"] = lr_decay;
        json["word_cutoff"] = word_cutoff;
        json["word_file"] = word_file;
        json["word_finetune"] = word_finetune;
        json["l2_reg"] = l2_reg;
        json["optimizer"] = static_cast<int>(optimizer);
        return json;
    }

    void fromJson(const Json::Value &json) override {
        word_dim = json["word_dim"].asInt();
        hidden_dim = json["hidden_dim"].asInt();
        hidden_layer = json["hidden_layer"].asInt();
        head_count = json["head_count"].asInt();
        dropout = json["dropout"].asFloat();
        batch_size = json["batch_size"].asInt();
        beam_size = json["beam_size"].asInt();
        learning_rate = json["learning_rate"].asFloat();
        lr_decay = json["lr_decay"].asBool();
        warm_up_iterations = json["warm_up_iterations"].asInt();
        word_cutoff = json["word_cutoff"].asInt();
        word_file = json["word_file"].asString();
        word_finetune = json["word_finetune"].asBool();
        l2_reg = json["l2_reg"].asFloat();
        optimizer = static_cast<Optimizer>(json["optimizer"].asInt());
    }

    void print() const {
        std::cout << "word_dim:" << word_dim << std::endl
            << "hidden_dim:" << hidden_dim << std::endl
            << "hidden_layer:" << hidden_layer << std::endl
            << "head_count:" << head_count << std::endl
            << "dropout:" << dropout << std::endl
            << "batch_size:" << batch_size << std::endl
            << "beam_size:" << beam_size << std::endl
            << "learning_rate:" << learning_rate << std::endl
            << "lr_decay:" << lr_decay << std::endl
            << "warm_up_iterations:" << warm_up_iterations << std::endl
	    << "word_cutoff:" << word_cutoff << std::endl
	    << "word_file:" << word_file << std::endl
    	    << "word_finetune:" << word_finetune << std::endl
    	    << "l2_reg:" << l2_reg << std::endl
    	    << "optimizer:" << optimizer << std::endl; 
    }
};

#endif
