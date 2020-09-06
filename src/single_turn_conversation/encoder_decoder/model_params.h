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
    LookupTable<SparseParam> encoder_lookup_table;
    LookupTable<Param> decoder_lookup_table;
    UniParams hidden_to_wordvector_params;
    LSTM1Params lstm_params;
    AdditiveAttentionParams attention_params;

    ModelParams() : encoder_lookup_table("encoder_lookup_table"),
    decoder_lookup_table("decoder_lookup_table"),
    hidden_to_wordvector_params("hidden_to_wordvector_params"), lstm_params("lstm_params"),
    attention_params("attention_params") {}

    Json::Value toJson() const override {
        Json::Value json;
        json["encoder_lookup_table"] = encoder_lookup_table.toJson();
        json["decoder_lookup_table"] = decoder_lookup_table.toJson();
        json["hidden_to_wordvector_params"] = hidden_to_wordvector_params.toJson();
        json["lstm_params"] = lstm_params.toJson();
        json["attention_params"] = attention_params.toJson();
        return json;
    }

    void fromJson(const Json::Value &json) override {
        encoder_lookup_table.fromJson(json["encoder_lookup_table"]);
        decoder_lookup_table.fromJson(json["decoder_lookup_table"]);
        hidden_to_wordvector_params.fromJson(json["hidden_to_wordvector_params"]);
        lstm_params.fromJson(json["lstm_params"]);
        attention_params.fromJson(json["attention_params"]);
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        return {&encoder_lookup_table, &decoder_lookup_table, &hidden_to_wordvector_params,
            &lstm_params, &lstm_params, &attention_params};
    }
#endif

protected:
    virtual std::vector<Tunable<BaseParam>*> tunableComponents() override {
        return {&encoder_lookup_table, &decoder_lookup_table, &hidden_to_wordvector_params,
            &lstm_params, &lstm_params, &attention_params};
    }
};

#endif
