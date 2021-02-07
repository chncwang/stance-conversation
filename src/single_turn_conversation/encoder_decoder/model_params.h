#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_MODEL_PARAMS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_MODEL_PARAMS_H

#include <fstream>
#include <iostream>

#include "N3LDG.h"

struct ModelParams : public N3LDGSerializable, public TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents
#endif
{
    LookupTable<Param> lookup_table;
    LookupTable<Param> lookup_table_scratch;
    UniParams hidden_to_wordvector_params;
    TransformerEncoderParams transformer_encoder_params;
    LayerNormalizationParams enc_norm;
    LayerNormalizationParams dec_norm;
    TransformerDecoderParams decoder_params;

    ModelParams() : lookup_table("lookup_table"),
    hidden_to_wordvector_params("hidden_to_wordvector_params"),
    transformer_encoder_params("encoder"), enc_norm("enc_norm"), dec_norm("dec_norm"),
    decoder_params("decoder_params") {}

    Json::Value toJson() const override {
        Json::Value json;
        json["lookup_table"] = lookup_table.toJson();
        json["lookup_table_scratch"] = lookup_table_scratch.toJson();
        json["hidden_to_wordvector_params"] = hidden_to_wordvector_params.toJson();
        json["transformer_encoder_params"] = transformer_encoder_params.toJson();
        json["enc_norm"] = enc_norm.toJson();
        json["dec_norm"] = dec_norm.toJson();
        json["decoder_params"] = decoder_params.toJson();
        return json;
    }

    void fromJson(const Json::Value &json) override {
        lookup_table.fromJson(json["lookup_table"]);
        lookup_table_scratch.fromJson(json["lookup_table_scratch"]);
        hidden_to_wordvector_params.fromJson(json["hidden_to_wordvector_params"]);
        transformer_encoder_params.fromJson(json["transformer_encoder_params"]);
        enc_norm.fromJson(json["enc_norm"]);
        dec_norm.fromJson(json["dec_norm"]);
        decoder_params.fromJson(json["decoder_params"]);
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        return {&lookup_table, &lookup_table_scratch, &hidden_to_wordvector_params,
            &transformer_encoder_params, &enc_norm, &dec_norm, &decoder_params};
    }
#endif

protected:
    virtual std::vector<Tunable<BaseParam>*> tunableComponents() override {
        return {&lookup_table,
//            &lookup_table_scratch,
            &hidden_to_wordvector_params, &transformer_encoder_params, &enc_norm, &dec_norm,
            &decoder_params};
    }
};

#endif
