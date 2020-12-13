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
    UniParams hidden_to_wordvector_params;
    TransformerEncoderParams transformer_encoder_params;
    BiasParam output_bias_params;
    Param begin_emb;
    Param hidden_embs;
    AdditiveAttentionParams attention_params;
    LSTM1Params decoder_params;

    ModelParams() : lookup_table("lookup_table"),
    hidden_to_wordvector_params("hidden_to_wordvector_params"),
    transformer_encoder_params("encoder"),
    output_bias_params("output_bias_params"), begin_emb("begin_emb"), hidden_embs("hidden_embs"),
    attention_params("attention_params"), decoder_params("decoder_params") {}

    Json::Value toJson() const override {
        Json::Value json;
        json["lookup_table"] = lookup_table.toJson();
        json["hidden_to_wordvector_params"] = hidden_to_wordvector_params.toJson();
        json["transformer_encoder_params"] = transformer_encoder_params.toJson();
        json["output_bias_params"] = output_bias_params.toJson();
        json["begin_emb"] = begin_emb.toJson();
        json["hidden_embs"] = hidden_embs.toJson();
        json["attention_params"] = attention_params.toJson();
        json["decoder_params"] = decoder_params.toJson();
        return json;
    }

    void fromJson(const Json::Value &json) override {
        lookup_table.fromJson(json["lookup_table"]);
        hidden_to_wordvector_params.fromJson(json["hidden_to_wordvector_params"]);
        transformer_encoder_params.fromJson(json["left_to_right_encoder_params"]);
        output_bias_params.fromJson(json["output_bias_params"]);
        begin_emb.fromJson(json["begin_emb"]);
        hidden_embs.fromJson(json["hidden_embs"]);
        attention_params.fromJson(json["attention_params"]);
        decoder_params.fromJson(json["decoder_params"]);
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        return {&lookup_table, &hidden_to_wordvector_params, &transformer_encoder_params,
            &output_bias_params, &begin_emb, &hidden_embs, &attention_params, &decoder_params};
    }
#endif

protected:
    virtual std::vector<Tunable<BaseParam>*> tunableComponents() override {
        return {&lookup_table, &hidden_to_wordvector_params, &transformer_encoder_params,
            &output_bias_params, &begin_emb, &hidden_embs, &attention_params, &decoder_params};
    }
};

#endif
