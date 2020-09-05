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
#include "single_turn_conversation/encoder_decoder/decoder_components.h"
#include "single_turn_conversation/def.h"

using namespace std;

set<std::array<int, 3>> triSet(const vector<int> &sentence) {
    if (sentence.size() < 3) {
        cerr << "triSet" << endl;
        abort();
    }
    using std::array;
    set<array<int, 3>> results;
    for (int i = 0; i < sentence.size() - 2; ++i) {
        array<int, 3> tri = {sentence.at(i + 0), sentence.at(i + 1), sentence.at(i + 2)};
        results.insert(tri);
    }

    if (results.size() != sentence.size() - 2) {
        cerr << boost::format("triSet - result size is %1%, but sentence len is %2%") %
            results.size() % sentence.size() << endl;
        abort();
    }

    return results;
}

set<int> repeatedIds(const vector<int> &sentence) {
    auto tri_set = triSet(sentence);
    set<int> results;
    for (const auto &tri : tri_set) {
        int sentence_len = sentence.size();
        if (tri.at(0) == sentence.at(sentence_len - 2) &&
                tri.at(1) == sentence.at(sentence_len - 1)) {
            results.insert(tri.at(2));
        }
    }
    return results;
}

struct WordIdAndProbability {
    int word_id;
    dtype probability;

    WordIdAndProbability() = default;
    WordIdAndProbability(const WordIdAndProbability &word_id_and_probability) = default;
    WordIdAndProbability(int wordid, dtype prob) : word_id(wordid), probability(prob) {}
};

string getSentence(const vector<int> &word_ids_vector, const ModelParams &model_params) {
    string words;
    for (const int &w : word_ids_vector) {
        string str = model_params.decoder_lookup_table.elems.from_id(w);
        words += str;
    }
    return words;
}

#define BEAM_SEARCH_KEY "beam_search"

class BeamSearchResult {
public:
    BeamSearchResult() {
        ngram_counts_ = {0, 0, 0};
    }
    BeamSearchResult(const BeamSearchResult &beam_search_result) = default;
    BeamSearchResult(const DecoderComponents &decoder_components,
            const vector<WordIdAndProbability> &pathh,
            dtype log_probability) : decoder_components_(decoder_components), path_(pathh),
            final_log_probability(log_probability) {
                ngram_counts_ = {0, 0, 0};
            }

    dtype finalScore() const {
//        set<int> unique_words;
//        for (const auto &p : path_) {
//            unique_words.insert(p.word_id);
//        }
//        for (int n = 2; n < 10; ++n) {
//            if (path_.size() >= n * 2) {
//                for (int i = path_.size() - n * 2; i>=0;--i) {
//                    bool ngram_hit = true;
//                    for (int j = 0; j < n; ++j) {
//                        if (path_.at(i + j).word_id != path_.at(path_.size() - n + j).word_id) {
//                            ngram_hit = false;
//                            break;
//                        }
//                    }
//                    if (ngram_hit) {
//                        return -1e10;
//                    }
//                }
//            }
//        }
        return (final_log_probability ) / path_.size();
//        return final_log_probability;
    }

    dtype finalLogProbability() const {
        return final_log_probability;
    }

    const vector<WordIdAndProbability> &getPath() const {
        return path_;
    }

    const DecoderComponents &decoderComponents() const {
        return decoder_components_;
    }

    void setExtraScore(dtype extra_score) {
        extra_score_ = extra_score;
    }

    dtype getExtraScore() const {
        return extra_score_;
    }

    const std::array<int, 3> &ngramCounts() const {
        return ngram_counts_;
    }

    void setNgramCounts(const std::array<int, 3> &counts) {
        ngram_counts_ = counts;
    }

private:
    DecoderComponents decoder_components_;
    vector<WordIdAndProbability> path_;
    dtype final_log_probability;
    dtype extra_score_;
    std::array<int, 3> ngram_counts_ = {};
};

void printWordIds(const vector<WordIdAndProbability> &word_ids_with_probability_vector,
        const LookupTable<Param> &lookup_table,
        bool print_space = false) {
    for (const WordIdAndProbability &ids : word_ids_with_probability_vector) {
        cout << lookup_table.elems.from_id(ids.word_id);
        if (print_space && &ids != &word_ids_with_probability_vector.back()) {
            cout << " ";
        }
    }
    cout << endl;
}

int countNgramDuplicate(const vector<int> &ids, int n) {
    if (n >= ids.size()) {
        return 0;
    }
    vector<int> target;
    for (int i = 0; i < n; ++i) {
        target.push_back(ids.at(ids.size() - n + i));
    }

    int duplicate_count = 0;

    for (int i = 0; i < ids.size() - n; ++i) {
        bool same = true;
        for (int j = 0; j < n; ++j) {
            if (target.at(j) != ids.at(i + j)) {
                same = false;
                break;
            }
        }
        if (same) {
            ++duplicate_count;
        }
    }

    return duplicate_count;
}

vector<BeamSearchResult> mostProbableResults(
        const vector<DecoderComponents> &beam,
        const vector<BeamSearchResult> &last_results,
        int current_word,
        int k,
        const ModelParams &model_params,
        const DefaultConfig &default_config,
        bool is_first,
        bool check_tri_gram,
        const vector<string> &black_list,
        set<int> &searched_word_ids,
        Graph &graph) {
    vector<Node *> nodes;
    for (const DecoderComponents &decoder_components : beam) {
        nodes.push_back(decoder_components.wordvector_to_onehots.at(current_word - 1));
        if (is_first) {
            break;
        }
    }
    if (!is_first && nodes.size() != last_results.size() && !last_results.empty()) {
        cerr << boost::format(
                "nodes size is not equal to last_results size, nodes size is %1% but last_results size is %2%")
            % nodes.size() % last_results.size() << endl;
        abort();
    }

    auto cmp = [&](const BeamSearchResult &a, const BeamSearchResult &b) {
        graph.addFLOPs(1, BEAM_SEARCH_KEY);
        return a.finalScore() > b.finalScore();
    };
    priority_queue<BeamSearchResult, vector<BeamSearchResult>, decltype(cmp)> queue(cmp);
    vector<BeamSearchResult> results;
    for (int i = 0; i < (is_first ? 1 : nodes.size()); ++i) {
        Node &node = *nodes.at(i);
#if USE_GPU
        node.val().initOnMemory(node.getDim());
        node.val().copyFromDeviceToHost();
#endif
        set<int> repeated_ids;
        if (check_tri_gram) {
            vector<int> word_ids;
            for (const auto &e : last_results.at(i).getPath()) {
                word_ids.push_back(e.word_id);
            }
            repeated_ids = repeatedIds(word_ids);
        }
        for (int j = 0; j < nodes.at(i)->getDim(); ++j) {
            if (repeated_ids.find(j) != repeated_ids.end()) {
                continue;
            }
            if (is_first) {
                if (searched_word_ids.find(j) != searched_word_ids.end()) {
                    cout << boost::format("word id searched:%1% word:%2%\n") % j %
                        model_params.decoder_lookup_table.elems.from_id(j);
                    continue;
                }
            }
            if (j == model_params.decoder_lookup_table.getElemId(::unknownkey)) {
                continue;
            }
            dtype word_probability = node.getVal().v[j];
            dtype log_probability = log(word_probability);
            graph.addFLOPs(1, BEAM_SEARCH_KEY);
            vector<WordIdAndProbability> word_ids;
            if (!last_results.empty()) {
                log_probability += last_results.at(i).finalLogProbability();
                graph.addFLOPs(1, BEAM_SEARCH_KEY);
                word_ids = last_results.at(i).getPath();
            }

            word_ids.push_back(WordIdAndProbability(j, word_probability));

            BeamSearchResult beam_search_result(beam.at(i), word_ids, log_probability);
            graph.addFLOPs(1, BEAM_SEARCH_KEY);

            if (queue.size() < k) {
                queue.push(beam_search_result);
            } else if (queue.top().finalScore() < beam_search_result.finalScore()) {
                graph.addFLOPs(1, BEAM_SEARCH_KEY);
                queue.pop();
                queue.push(beam_search_result);
            } else {
                graph.addFLOPs(1, BEAM_SEARCH_KEY);
            }
        }
    }

    cout << "queue size:" << queue.size() << endl;
    while (!queue.empty()) {
        auto &e = queue.top();
        if (is_first) {
            int size = e.getPath().size();
            if (size != 1) {
                cerr << boost::format("size is not 1:%1%\n") % size;
                abort();
            }
            searched_word_ids.insert(e.getPath().at(0).word_id);
        }
        results.push_back(e);
        queue.pop();
    }

    vector<BeamSearchResult> final_results;
    int i = 0;
    for (const BeamSearchResult &result : results) {
        vector<int> ids = transferVector<int, WordIdAndProbability>(result.getPath(),
                [](const WordIdAndProbability &in) ->int {return in.word_id;});
        string sentence = ::getSentence(ids, model_params);
        bool contain_black = false;
        for (const string str : black_list) {
            utf8_string utf8_str(str), utf8_sentece(sentence);
            if (utf8_sentece.find(utf8_str) != string::npos) {
                contain_black = true;
                break;
            }
        }
        if (contain_black) {
            continue;
        }
        final_results.push_back(result);
        cout << boost::format("mostProbableResults - i:%1% prob:%2% score:%3%") % i %
            result.finalLogProbability() % result.finalScore() << endl;
        printWordIds(result.getPath(), model_params.decoder_lookup_table);
        ++i;
    }

    return final_results;
}

struct GraphBuilder {
    vector<Node *> encoder_lookups;
    DynamicLSTMBuilder left_to_right_encoder;

    void forward(Graph &graph, const vector<string> &sentence,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            bool is_training) {
        using namespace n3ldg_plus;
        Node *hidden_bucket = bucket(graph, hyper_params.hidden_dim, 0);
        for (const string &word : sentence) {
            Node *input_lookup = embedding(graph, model_params.encoder_lookup_table, word);
            Node *dropout_node = dropout(graph, *input_lookup, hyper_params.dropout, is_training);
            encoder_lookups.push_back(dropout_node);
        }

        for (Node* node : encoder_lookups) {
            left_to_right_encoder.forward(graph, model_params.left_to_right_encoder_params, *node,
                    *hidden_bucket, *hidden_bucket, hyper_params.dropout, is_training);
        }
    }

    void forwardDecoder(Graph &graph, DecoderComponents &decoder_components,
            const vector<string> &answer,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            bool is_training) {
        for (int i = 0; i < answer.size(); ++i) {
            forwardDecoderByOneStep(graph, decoder_components, i, i == 0 ? nullptr :
                    &answer.at(i - 1), hyper_params, model_params, is_training);
        }
    }

    void forwardDecoderByOneStep(Graph &graph, DecoderComponents &decoder_components, int i,
            const string *answer,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            bool is_training) {
        using namespace n3ldg_plus;
        Node *last_input;
        if (i > 0) {
            Node *decoder_lookup = embedding(graph, model_params.decoder_lookup_table, *answer);
            decoder_lookup = dropout(graph, *decoder_lookup, hyper_params.dropout, is_training);
            decoder_components.decoder_lookups.push_back(decoder_lookup);
            last_input = decoder_components.decoder_lookups.at(i - 1);
        } else {
            last_input = bucket(graph, hyper_params.word_dim, 0);
        }

        decoder_components.forward(graph, hyper_params, model_params, *last_input,
                left_to_right_encoder._hiddens, is_training);

        Node *decoder_to_wordvector = decoder_components.decoderToWordVectors(graph, hyper_params,
                model_params, i);
        decoder_components.decoder_to_wordvectors.push_back(decoder_to_wordvector);
        Node *wordvector_to_onehot = linearWordVector(graph,
                model_params.decoder_lookup_table.nVSize, model_params.decoder_lookup_table.E,
                *decoder_to_wordvector);
        Node *softmax = n3ldg_plus::softmax(graph, *wordvector_to_onehot);
        decoder_components.wordvector_to_onehots.push_back(softmax);
    }

    pair<vector<WordIdAndProbability>, dtype> forwardDecoderUsingBeamSearch(Graph &graph,
            const vector<DecoderComponents> &decoder_components_beam,
            int k,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            const DefaultConfig &default_config,
            const vector<string> &black_list) {
        vector<pair<vector<WordIdAndProbability>, dtype>> word_ids_result;
        vector<BeamSearchResult> most_probable_results;
        vector<string> last_answers;
        bool succeeded = false;
        set<int> searched_word_ids;

        for (int iter = 0; ; ++iter) {
            cout << boost::format("forwardDecoderUsingBeamSearch iter:%1%\n") % iter;
            most_probable_results.clear();
            auto beam = decoder_components_beam;
            cout << boost::format("beam size:%1%\n") % beam.size();

            int ended_count = word_ids_result.size();
            if (ended_count >= default_config.result_count_factor * k) {
                break;
            }

            for (int i = 0;; ++i) {
                cout << boost::format("forwardDecoderUsingBeamSearch i:%1%\n") % i;
                if (word_ids_result.size() >= default_config.result_count_factor * k ||
                        i > default_config.cut_length) {
                    break;
                }

                last_answers.clear();
                if (i > 0) {
                    most_probable_results = mostProbableResults(beam, most_probable_results, i,
                            k, model_params, default_config, i == 1, i >= 4, black_list,
                            searched_word_ids, graph);
                    cout << boost::format("most_probable_results size:%1%") %
                        most_probable_results.size() << endl;
                    auto last_beam = beam;
                    beam.clear();
                    vector<BeamSearchResult> stop_removed_results;
                    int j = 0;
                    for (BeamSearchResult &beam_search_result : most_probable_results) {
                        const vector<WordIdAndProbability> &word_ids =
                            beam_search_result.getPath();

                        int last_word_id = word_ids.at(word_ids.size() - 1).word_id;
                        const string &word = model_params.decoder_lookup_table.elems.from_id(
                                last_word_id);
                        if (word == STOP_SYMBOL) {
                            word_ids_result.push_back(make_pair(word_ids,
                                        beam_search_result.finalScore()));
                            succeeded = word == STOP_SYMBOL;
                        } else {
                            stop_removed_results.push_back(beam_search_result);
                            last_answers.push_back(word);
                            beam.push_back(beam_search_result.decoderComponents());
                        }
                        ++j;
                    }
                    most_probable_results = stop_removed_results;
                }

                if (beam.empty()) {
                    cout << boost::format("break for beam empty\n");
                    break;
                }

                for (int beam_i = 0; beam_i < beam.size(); ++beam_i) {
                    DecoderComponents &decoder_components = beam.at(beam_i);
                    forwardDecoderByOneStep(graph, decoder_components, i,
                            i == 0 ? nullptr : &last_answers.at(beam_i), hyper_params,
                            model_params, false);
                    if (i == 0) {
                        break;
                    }
                }

                graph.compute();
            }
        }

        if (word_ids_result.size() < default_config.result_count_factor * k) {
            cerr << boost::format("word_ids_result size is %1%, but beam_size is %2%") %
                word_ids_result.size() % k << endl;
            abort();
        }

        for (const auto &pair : word_ids_result) {
            const vector<WordIdAndProbability> ids = pair.first;
            cout << boost::format("beam result:%1%") % exp(pair.second) << endl;
            printWordIds(ids, model_params.decoder_lookup_table);
        }

        auto compair = [](const pair<vector<WordIdAndProbability>, dtype> &a,
                const pair<vector<WordIdAndProbability>, dtype> &b) {
            return a.second < b.second;
        };
        auto max = max_element(word_ids_result.begin(), word_ids_result.end(), compair);

        return make_pair(max->first, exp(max->second));
    }
};

#endif
