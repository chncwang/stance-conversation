#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H

#include <memory>
#include "n3ldg-plus/n3ldg-plus.h"
#include "single_turn_conversation/encoder_decoder/model_params.h"
#include "single_turn_conversation/encoder_decoder/hyper_params.h"

using namespace n3ldg_plus;

Node* decoderToWordVectors(const std::vector<Node *> &hiddens, int dec_sentence_len,
        const HyperParams &hyper_params,
        ModelParams &model_params) {
    using namespace n3ldg_plus;
    Node *hidden = concatToMatrix(hiddens);
    Node *decoder_to_wordvector = n3ldg_plus::linear(*hidden,
            model_params.hidden_to_wordvector_params);
    return decoder_to_wordvector;
}

struct DecoderCellComponents {
    Node *wordvector_to_onehot;
    LSTMState state;

    DecoderCellComponents() = default;

    Node* decoderToWordVectors(const HyperParams &hyper_params, ModelParams &model_params) {
        using namespace n3ldg_plus;
        Node *decoder_to_wordvector = n3ldg_plus::linear(*state.hidden,
                model_params.hidden_to_wordvector_params);
        return decoder_to_wordvector;
    }
};
#endif
