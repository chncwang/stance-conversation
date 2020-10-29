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
    Param stance_embeddings;
    UniParams hidden_to_wordvector_params;
    LSTM1Params left_to_right_encoder_params;
    LSTM1Params left_to_right_decoder_params;
    LSTM1Params sel_encoder_params;
    AdditiveAttentionParams attention_params;
    AdditiveAttentionParams attention_params_for_sel;

    ModelParams() : lookup_table("lookup_table"), stance_embeddings("stance_embeddings"),
    hidden_to_wordvector_params("hidden_to_wordvector_params"),
    left_to_right_encoder_params("encoder"), left_to_right_decoder_params("decoder"),
    sel_encoder_params("sel_encoder_params"), attention_params("attention_params"),
    attention_params_for_sel("attention_params_for_sel") {}

    Json::Value toJson() const override {
        Json::Value json;
        json["lookup_table"] = lookup_table.toJson();
        json["stance_embeddings"] = lookup_table.toJson();
        json["hidden_to_wordvector_params"] = hidden_to_wordvector_params.toJson();
        json["left_to_right_encoder_params"] = left_to_right_encoder_params.toJson();
        json["left_to_right_decoder_params"] = left_to_right_decoder_params.toJson();
        json["sel_encoder_params"] = sel_encoder_params.toJson();
        json["attention_params"] = attention_params.toJson();
        json["attention_params_for_sel"] = attention_params_for_sel.toJson();
        return json;
    }

    void fromJson(const Json::Value &json) override {
        lookup_table.fromJson(json["lookup_table"]);
        lookup_table.fromJson(json["stance_embeddings"]);
        hidden_to_wordvector_params.fromJson(json["hidden_to_wordvector_params"]);
        left_to_right_encoder_params.fromJson(json["left_to_right_encoder_params"]);
        left_to_right_decoder_params.fromJson(json["left_to_right_decoder_params"]);
        sel_encoder_params.fromJson(json["sel_encoder_params"]);
        attention_params.fromJson(json["attention_params"]);
        attention_params_for_sel.fromJson(json["attention_params_for_sel"]);
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        return {&lookup_table, &stance_embeddings, &hidden_to_wordvector_params,
            &left_to_right_encoder_params, &left_to_right_decoder_params, &sel_encoder_params,
            &attention_params, &attention_params_for_sel};
    }
#endif

protected:
    virtual std::vector<Tunable<BaseParam>*> tunableComponents() override {
        return {&lookup_table, &stance_embeddings, &hidden_to_wordvector_params,
            &left_to_right_encoder_params, &left_to_right_decoder_params, &attention_params,
            &attention_params_for_sel};
    }
};

struct ResSelModelParams : public N3LDGSerializable, public TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents
#endif
{
    LookupTable<Param> lookup_table;
    Param stance_embeddings;
    LSTM1Params left_to_right_encoder_params;
    LSTM1Params response_encoder_params;
    UniParams post_rep_params;
    UniParams response_rep_params;

    ResSelModelParams() : lookup_table("lookup_table"), stance_embeddings("stance_embeddings"),
    left_to_right_encoder_params("encoder"), response_encoder_params("response_encoder_params"),
    post_rep_params("post_rep_params"), response_rep_params("response_rep_params") {}

    Json::Value toJson() const override {
        Json::Value json;
        json["lookup_table"] = lookup_table.toJson();
        json["stance_embeddings"] = stance_embeddings.toJson();
        json["left_to_right_encoder_params"] = left_to_right_encoder_params.toJson();
        json["response_encoder_params"] = response_encoder_params.toJson();
        json["post_rep_params"] = post_rep_params.toJson();
        json["response_rep_params"] = response_rep_params.toJson();
        return json;
    }

    void fromJson(const Json::Value &json) override {
        lookup_table.fromJson(json["lookup_table"]);
        stance_embeddings.fromJson(json["stance_embeddings"]);
        left_to_right_encoder_params.fromJson(json["left_to_right_encoder_params"]);
        response_encoder_params.fromJson(json["response_encoder_params"]);
        post_rep_params.fromJson(json["post_rep_params"]);
        response_rep_params.fromJson(json["response_rep_params"]);
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        return {&lookup_table, &stance_embeddings, &left_to_right_encoder_params,
            &response_encoder_params, &post_rep_params, &response_rep_params};
    }
#endif

protected:
    virtual std::vector<Tunable<BaseParam>*> tunableComponents() override {
        return {&lookup_table, &stance_embeddings, &left_to_right_encoder_params,
            &response_encoder_params, &post_rep_params, &response_rep_params};
    }
};


#endif
