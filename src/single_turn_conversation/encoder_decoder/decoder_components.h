#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H

#include <memory>
#include "n3ldg-plus/n3ldg-plus.h"
#include "single_turn_conversation/encoder_decoder/model_params.h"
#include "single_turn_conversation/encoder_decoder/hyper_params.h"

using namespace n3ldg_plus;

struct DecoderComponents {
    Node *wordvector_to_onehots;
    TransformerDecoderBuilder decoder;

    DecoderComponents(Graph &graph, TransformerDecoderParams &params, Node &encoder_hiddens,
            int src_sentence_len, dtype dropout, bool is_training) : decoder(graph, params,
                encoder_hiddens, src_sentence_len, dropout, is_training) {}

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

struct DecoderCellComponents {
    Node *wordvector_to_onehot;
    n3ldg_plus::TransformerDecoderCellBuilder decoder;

    DecoderCellComponents(Graph &graph, TransformerDecoderParams &params, Node &encoder_hiddens,
            int src_sentence_len, dtype dropout, bool is_training) : decoder(graph, params,
                encoder_hiddens, src_sentence_len, dropout, is_training) {}

    Node* decoderToWordVectors(Graph &graph, const HyperParams &hyper_params,
            ModelParams &model_params) {
        using namespace n3ldg_plus;
        Node *normed = layerNormalization(graph, model_params.dec_norm,
                *decoder.hiddenLayers().back().back());
        Node *decoder_to_wordvector = n3ldg_plus::linear(graph, *normed,
                model_params.hidden_to_wordvector_params);
        return decoder_to_wordvector;
    }
};
#endif
