#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_MODEL_PARAMS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_MODEL_PARAMS_H

#include <fstream>
#include <iostream>

#include "n3ldg-plus/n3ldg-plus.h"

using namespace ::n3ldg_plus;

struct ModelParams : public TunableParamCollection
#if USE_GPU
, public cuda::TransferableComponents
#endif
{
    Embedding<Param> lookup_table;
    TransformerEncoderParams transformer_encoder_params;
    TransformerDecoderParams decoder_params;

    ModelParams() : transformer_encoder_params("encoder"), decoder_params("decoder_params") {}


    template<typename Archive>
    void serialize(Archive &ar) {
        ar(lookup_table, transformer_encoder_params, decoder_params);
    }

#if USE_GPU
    std::vector<cuda::Transferable *> transferablePtrs() override {
        return {&lookup_table, &transformer_encoder_params, &decoder_params};
    }
#endif

protected:
    virtual std::vector<TunableParam *> tunableComponents() override {
        return {&lookup_table, &transformer_encoder_params, &decoder_params};
    }
};

#endif
