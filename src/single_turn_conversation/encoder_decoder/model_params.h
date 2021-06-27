#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_MODEL_PARAMS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_MODEL_PARAMS_H

#include <fstream>
#include <iostream>

#include "insnet/insnet.h"

using namespace insnet;

struct ModelParams : public TunableParamCollection
#if USE_GPU
, public cuda::TransferableComponents
#endif
{
    Embedding<Param> lookup_table;
    TransformerEncoderParams transformer_encoder_params;
    TransformerDecoderParams decoder_params;
    BiasParam output_bias;

    ModelParams() : transformer_encoder_params("encoder"), decoder_params("decoder_params"),
    output_bias("output_bias") {}


    template<typename Archive>
    void serialize(Archive &ar) {
        ar(lookup_table, transformer_encoder_params, decoder_params, output_bias);
    }

#if USE_GPU
    std::vector<cuda::Transferable *> transferablePtrs() override {
        return {&lookup_table, &transformer_encoder_params, &decoder_params, &output_bias};
    }
#endif

protected:
    virtual std::vector<TunableParam *> tunableComponents() override {
        return {&lookup_table, &transformer_encoder_params, &decoder_params, &output_bias};
    }
};

#endif
