#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H

#include <memory>
#include "N3LDG.h"
#include "single_turn_conversation/encoder_decoder/model_params.h"
#include "single_turn_conversation/encoder_decoder/hyper_params.h"

struct DecoderComponents {
    std::vector<Node *> decoder_lookups;
    std::vector<Node *> wordvector_to_onehots;
    DynamicLSTMBuilder decoder;
    vector<Node *> contexts;

    void forward(Graph &graph, const HyperParams &hyper_params, ModelParams &model_params,
            Node &input,
            vector<Node *> &encoder_hiddens,
            bool is_traning) {
        AdditiveAttentionBuilder attention_builder;
        Node *guide;
        if (decoder.size() == 0) {
            guide = n3ldg_plus::embedding(graph, model_params.hidden_embs, 0);
        } else {
            guide = decoder._hiddens.back();
        }
        attention_builder.forward(graph, model_params.attention_params, encoder_hiddens, *guide);
        contexts.push_back(attention_builder._hidden);
        vector<Node *> ins = {&input, attention_builder._hidden};
        Node *concat = n3ldg_plus::concat(graph, ins);
        decoder.forward(graph, model_params.decoder_params, *concat,
                *n3ldg_plus::embedding(graph, model_params.hidden_embs, 0),
                *n3ldg_plus::embedding(graph, model_params.hidden_embs, 1),
                hyper_params.dropout, is_traning);
    }

    Node* decoderToWordVectors(Graph &graph, const HyperParams &hyper_params,
            ModelParams &model_params,
            int i) {
        using namespace n3ldg_plus;
        vector<Node *> concat_inputs = {decoder._hiddens.at(i), contexts.at(i),
            static_cast<Node*>(decoder_lookups.at(i))};
        Node *concat_node = concat(graph, concat_inputs);

        Node *decoder_to_wordvector = n3ldg_plus::linear(graph,
                model_params.hidden_to_wordvector_params, *concat_node);
        return decoder_to_wordvector;
    }
};

#endif
