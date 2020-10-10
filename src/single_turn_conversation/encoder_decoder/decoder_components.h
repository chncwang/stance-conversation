#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H

#include <memory>
#include "N3LDG.h"
#include "single_turn_conversation/encoder_decoder/model_params.h"
#include "single_turn_conversation/encoder_decoder/hyper_params.h"

struct DecoderComponents {
    std::vector<Node *> decoder_lookups;
    std::vector<Node *> decoder_to_wordvectors;
    std::vector<Node *> wordvector_to_onehots;
    n3ldg_plus::TransformerDecoderBuilder decoder;

    DecoderComponents(Graph &graph, TransformerDecoderParams &params,
            const vector<Node *> &encoder_hiddens,
            dtype dropout,
            bool is_training) : decoder(graph, params, encoder_hiddens, dropout, is_training) {}

    void forward(vector<Node*> &input) {
        decoder.forward(input);
    }

    Node* decoderToWordVectors(Graph &graph, const HyperParams &hyper_params,
            ModelParams &model_params,
            int i) {
        using namespace n3ldg_plus;
//        vector<Node *> concat_inputs = {decoder.hiddenLayers().back().at(i),
//            i == 0 ? bucket(graph, hyper_params.word_dim, 0) :
//                static_cast<Node*>(decoder_lookups.at(i - 1))};
//        Node *concat_node = concat(graph, concat_inputs);

        Node *decoder_to_wordvector = n3ldg_plus::linear(graph,
                model_params.hidden_to_wordvector_params, *decoder.hiddenLayers().back().at(i));
        return decoder_to_wordvector;
    }
};

#endif
