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
            int src_sentence_len, dtype dropout) : decoder(params,
                encoder_hiddens, dropout) {}

    Node* decoderToWordVectors(Graph &graph, int dec_sentence_len,
            const HyperParams &hyper_params,
            ModelParams &model_params) {
        return decoder.hiddenLayers().back();
    }
};

struct DecoderCellComponents {
    Node *wordvector_to_onehot;
    n3ldg_plus::TransformerDecoderCellBuilder decoder;

    DecoderCellComponents(Graph &graph, TransformerDecoderParams &params, Node &encoder_hiddens,
            int src_sentence_len, dtype dropout) : decoder(params,
                encoder_hiddens, dropout) {}

    Node* decoderToWordVectors(const HyperParams &hyper_params, ModelParams &model_params) {
        return decoder.hiddenLayers().back().back();
    }
};
#endif
