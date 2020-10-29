#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_SEL_GRAPH_BUILDER_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_SEL_GRAPH_BUILDER_H

#include <cmath>
#include <vector>
#include <array>
#include <set>
#include <string>
#include <memory>
#include <tuple>
#include <queue>
#include <algorithm>
#include <boost/format.hpp>
#include "N3LDG.h"
#include "tinyutf8.h"
#include "model_params.h"
#include "hyper_params.h"
#include "single_turn_conversation/default_config.h"
#include "single_turn_conversation/def.h"
#include "single_turn_conversation//conversation_structure.h"

using namespace std;

Node *sentenceRep(Graph &graph, const vector<string> &sentence,
        ResSelModelParams &model_params,
        LSTM1Params &lstm_params,
        UniParams &rep_params,
        StanceCategory *stance_category) {
    using namespace n3ldg_plus;
    DynamicLSTMBuilder left_to_right_encoder;
    Node *hidden_bucket = bucket(graph, 1024, 0);
    for (const string &word : sentence) {
        Node *input_lookup = embedding(graph, model_params.lookup_table, word);
        Node *dropout_node = dropout(graph, *input_lookup, 0.1, false);
        Node *in = dropout_node;
        if (stance_category != nullptr) {
            Node *embedding = n3ldg_plus::embedding(graph, model_params.stance_embeddings,
                    static_cast<int>(*stance_category));
            in = n3ldg_plus::concat(graph, {dropout_node, embedding});
        }
        left_to_right_encoder.forward(graph, lstm_params, *in, *hidden_bucket, *hidden_bucket,
                0.1, false);
    }
    Node *rep = maxPool(graph, left_to_right_encoder._hiddens);
    rep = n3ldg_plus::linear(graph, rep_params, *rep);
    return rep;
}

vector<Node *> selectionProbs(Graph &graph, const vector<Node *> &post_reps,
        const vector<Node *> &res_reps) {
    Node *res_matrix = n3ldg_plus::concatToMatrix(graph, res_reps);
    vector<Node *> results;
    for (Node *p : post_reps) {
        Node *prob = n3ldg_plus::dotAttentionWeights(graph, *res_matrix, *p);
        results.push_back(prob);
    }
    return results;
}

#endif
