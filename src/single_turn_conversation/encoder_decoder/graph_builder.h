#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_GRAPH_BUILDER_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_GRAPH_BUILDER_H

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

string getSentence(const vector<int> &word_ids_vector, const ModelParams &model_params) {
    string words;
    for (const int &w : word_ids_vector) {
        string str = model_params.lookup_table.elems.from_id(w);
        words += str;
    }
    return words;
}

Node *sentenceRep(Graph &graph, const vector<string> &sentence,
        const HyperParams &hyper_params,
        ModelParams &model_params,
        LSTM1Params &lstm_params,
        UniParams &rep_params,
        StanceCategory *stance_category,
        bool is_training) {
    using namespace n3ldg_plus;
    DynamicLSTMBuilder left_to_right_encoder;
    Node *hidden_bucket = bucket(graph, hyper_params.hidden_dim, 0);
    for (const string &word : sentence) {
        Node *input_lookup = embedding(graph, model_params.lookup_table, word);
        Node *dropout_node = dropout(graph, *input_lookup, hyper_params.dropout, is_training);
        Node *in = dropout_node;
        if (stance_category != nullptr) {
            Node *embedding = n3ldg_plus::embedding(graph, model_params.stance_embeddings,
                    static_cast<int>(*stance_category));
            in = n3ldg_plus::concat(graph, {dropout_node, embedding});
        }
        left_to_right_encoder.forward(graph, lstm_params, *in, *hidden_bucket, *hidden_bucket,
                hyper_params.dropout, is_training);
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
//    vector<Node *> inner_rs;
//    for (Node *r : res_reps) {
//        inner_rs.push_back(r);
//    }
//    for (Node *p : post_reps) {
//        auto r = n3ldg_plus::additiveAttentionWeights(graph, params, inner_rs, *p);
//        Node *m = n3ldg_plus::scalarConcat(graph, r);
//        m = n3ldg_plus::softmax(graph, *m);
//        results.push_back(m);
//    }
    return results;
}

#endif
