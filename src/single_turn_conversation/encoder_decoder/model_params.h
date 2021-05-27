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
    LinearParams hidden_to_wordvector_params;
    TransformerEncoderParams transformer_encoder_params;
    LayerNormParams enc_norm;
    LayerNormParams dec_norm;
    TransformerDecoderParams decoder_params;

    ModelParams() : hidden_to_wordvector_params("hidden_to_wordvector_params"),
    transformer_encoder_params("encoder"), enc_norm("enc_norm"), dec_norm("dec_norm"),
    decoder_params("decoder_params") {}


    template<typename Archive>
    void serialize(Archive &ar) {
        ar(lookup_table, hidden_to_wordvector_params, transformer_encoder_params, enc_norm,
                dec_norm, decoder_params);
    }

#if USE_GPU
    std::vector<cuda::Transferable *> transferablePtrs() override {
        return {&lookup_table, &hidden_to_wordvector_params, &transformer_encoder_params,
            &enc_norm, &dec_norm, &decoder_params};
    }
#endif

protected:
    virtual std::vector<TunableParam *> tunableComponents() override {
        return {&lookup_table, &hidden_to_wordvector_params, &transformer_encoder_params,
            &enc_norm, &dec_norm, &decoder_params};
    }
};

#endif
