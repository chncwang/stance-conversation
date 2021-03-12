#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H

#include <memory>
#include "N3LDG.h"
#include "single_turn_conversation/encoder_decoder/model_params.h"
#include "single_turn_conversation/encoder_decoder/hyper_params.h"

struct DecoderComponents {
    Node *wordvector_to_onehots;
    n3ldg_plus::TransformerDecoderBuilder decoder;

    DecoderComponents(Graph &graph, TransformerDecoderParams &params, Node &encoder_hiddens,
            int src_sentence_len, dtype dropout, bool is_training) : decoder(graph, params,
                encoder_hiddens, src_sentence_len, dropout, is_training) {}

    Node* decoderToWordVectors(Graph &graph, const HyperParams &hyper_params,
            ModelParams &model_params,
            int i) {
        using namespace n3ldg_plus;
        Node *normed = layerNormalization(graph, model_params.dec_norm,
                *decoder.hiddenLayers().back()->batch().at(i));
        Node *decoder_to_wordvector = n3ldg_plus::linear(graph, *normed,
                model_params.hidden_to_wordvector_params);
        return decoder_to_wordvector;
    }

    Node* decoderToWordVectors(Graph &graph, int dec_sentence_len,
            const HyperParams &hyper_params,
            ModelParams &model_params) {
        using namespace n3ldg_plus;
        Node *normed = layerNormalization(graph, model_params.dec_norm,
                *decoder.hiddenLayers().back(), dec_sentence_len);
        Node *decoder_to_wordvector = n3ldg_plus::linear(graph, *normed,
                model_params.hidden_to_wordvector_params);
        return decoder_to_wordvector;
    }
};

#endif
