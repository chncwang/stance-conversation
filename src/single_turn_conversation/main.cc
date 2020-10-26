#include "cxxopts.hpp"
#include <unistd.h>
#include <chrono>
#include <algorithm>
#include <random>
#include "INIReader.h"
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <iomanip>
#include <array>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <mutex>
#include <atomic>
#include <boost/format.hpp>
#include <boost/asio.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include <boost/algorithm/string/regex.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include "N3LDG.h"
#include "single_turn_conversation/data_manager.h"
#include "single_turn_conversation/def.h"
#include "single_turn_conversation/bleu.h"
#include "single_turn_conversation/perplex.h"
#include "single_turn_conversation/default_config.h"
#include "single_turn_conversation/encoder_decoder/graph_builder.h"
#include "single_turn_conversation/encoder_decoder/hyper_params.h"
#include "single_turn_conversation/encoder_decoder/model_params.h"

using namespace std;
using namespace cxxopts;
using namespace boost::asio;
using boost::is_any_of;
using boost::format;
using boost::filesystem::path;
using boost::filesystem::is_directory;
using boost::filesystem::directory_iterator;

void addWord(unordered_map<string, int> &word_counts, const string &word) {
    auto it = word_counts.find(word);
    if (it == word_counts.end()) {
        word_counts.insert(make_pair(word, 1));
    } else {
        it->second++;
    }
}

void addWord(unordered_map<string, int> &word_counts, const vector<string> &sentence) {
    for (const string &word : sentence) {
        addWord(word_counts, word);
    }
}

DefaultConfig parseDefaultConfig(INIReader &ini_reader) {
    DefaultConfig default_config;
    static const string SECTION = "default";
    default_config.train_pair_file = ini_reader.Get(SECTION, "train_pair_file", "");
    if (default_config.train_pair_file.empty()) {
        cerr << "pair file empty" << endl;
        abort();
    }

    default_config.dev_pair_file = ini_reader.Get(SECTION, "dev_pair_file", "");
    if (default_config.dev_pair_file.empty()) {
        cerr << "pair file empty" << endl;
        abort();
    }

    default_config.test_pair_file = ini_reader.Get(SECTION, "test_pair_file", "");
    if (default_config.test_pair_file.empty()) {
        cerr << "pair file empty" << endl;
        abort();
    }

    default_config.post_file = ini_reader.Get(SECTION, "post_file", "");
    if (default_config.post_file.empty()) {
        cerr << "post file empty" << endl;
        abort();
    }

    default_config.response_file = ini_reader.Get(SECTION, "response_file", "");
    if (default_config.post_file.empty()) {
        cerr << "post file empty" << endl;
        abort();
    }

    default_config.decoded_file = ini_reader.Get(SECTION, "decoded_file", "");

    string program_mode_str = ini_reader.Get(SECTION, "program_mode", "");
    ProgramMode program_mode;
    if (program_mode_str == "interacting") {
        program_mode = ProgramMode::INTERACTING;
    } else if (program_mode_str == "training") {
        program_mode = ProgramMode::TRAINING;
    } else if (program_mode_str == "decoding") {
        program_mode = ProgramMode::DECODING;
    } else if (program_mode_str == "metric") {
        program_mode = ProgramMode::METRIC;
    } else if (program_mode_str == "decoded_ppl") {
        program_mode = ProgramMode::DECODED_PPL;
    }
    else {
        cout << format("program mode is %1%") % program_mode_str << endl;
        abort();
    }
    default_config.program_mode = program_mode;

    default_config.check_grad = ini_reader.GetBoolean(SECTION, "check_grad", false);
    default_config.one_response = ini_reader.GetBoolean(SECTION, "one_response", false);
    default_config.learn_test = ini_reader.GetBoolean(SECTION, "learn_test", false);
    default_config.save_model_per_batch = ini_reader.GetBoolean(SECTION, "save_model_per_batch",
            false);
    default_config.split_unknown_words = ini_reader.GetBoolean(SECTION, "split_unknown_words",
            true);

    default_config.train_sample_count = ini_reader.GetInteger(SECTION, "train_sample_count",
            1000000000);
    default_config.dev_sample_count = ini_reader.GetInteger(SECTION, "dev_sample_count",
            1000000000);
    default_config.test_sample_count = ini_reader.GetInteger(SECTION, "test_sample_count",
            1000000000);
    default_config.hold_batch_size = ini_reader.GetInteger(SECTION, "hold_batch_size", 100);
    default_config.device_id = ini_reader.GetInteger(SECTION, "device_id", 0);
    default_config.seed = ini_reader.GetInteger(SECTION, "seed", 0);
    default_config.cut_length = ini_reader.GetInteger(SECTION, "cut_length", 30);
    default_config.max_epoch = ini_reader.GetInteger(SECTION, "max_epoch", 100);
    default_config.output_model_file_prefix = ini_reader.Get(SECTION, "output_model_file_prefix",
            "");
    default_config.input_model_file = ini_reader.Get(SECTION, "input_model_file", "");
    default_config.input_model_dir = ini_reader.Get(SECTION, "input_model_dir", "");
    default_config.black_list_file = ini_reader.Get(SECTION, "black_list_file", "");
    default_config.memory_in_gb = ini_reader.GetReal(SECTION, "memory_in_gb", 0.0f);
    default_config.result_count_factor = ini_reader.GetReal(SECTION, "result_count_factor", 1.0f);

    default_config.human_stance_file = ini_reader.Get(SECTION, "human_stance_file", "");
    default_config.auto_stance_file = ini_reader.Get(SECTION, "auto_stance_file", "");

    return default_config;
}

HyperParams parseHyperParams(INIReader &ini_reader) {
    HyperParams hyper_params;

    int word_dim = ini_reader.GetInteger("hyper", "word_dim", 0);
    if (word_dim <= 0) {
        cerr << "word_dim wrong" << endl;
        abort();
    }
    hyper_params.word_dim = word_dim;

    int stance_dim = ini_reader.GetInteger("hyper", "stance_dim", 0);
    if (stance_dim <= 0) {
        cerr << "stance_dim wrong" << endl;
        abort();
    }
    hyper_params.stance_dim = stance_dim;

    int encoding_hidden_dim = ini_reader.GetInteger("hyper", "hidden_dim", 0);
    if (encoding_hidden_dim <= 0) {
        cerr << "hidden_dim wrong" << endl;
        abort();
    }
    hyper_params.hidden_dim = encoding_hidden_dim;

    float dropout = ini_reader.GetReal("hyper", "dropout", 0.0);
    if (dropout < -1.0f || dropout >=1.0f) {
        cerr << "dropout wrong" << endl;
        abort();
    }
    hyper_params.dropout = dropout;

    int batch_size = ini_reader.GetInteger("hyper", "batch_size", 0);
    if (batch_size == 0) {
        cerr << "batch_size not found" << endl;
        abort();
    }
    hyper_params.batch_size = batch_size;

    int beam_size = ini_reader.GetInteger("hyper", "beam_size", 0);
    if (beam_size == 0) {
        cerr << "beam_size not found" << endl;
        abort();
    }
    hyper_params.beam_size = beam_size;

    float learning_rate = ini_reader.GetReal("hyper", "learning_rate", 0.001f);
    if (learning_rate <= 0.0f) {
        cerr << "learning_rate wrong" << endl;
        abort();
    }
    hyper_params.learning_rate = learning_rate;

    float min_learning_rate = ini_reader.GetReal("hyper", "min_learning_rate", 0.0001f);
    if (min_learning_rate < 0.0f) {
        cerr << "min_learning_rate wrong" << endl;
        abort();
    }
    hyper_params.min_learning_rate = min_learning_rate;

    float learning_rate_decay = ini_reader.GetReal("hyper", "learning_rate_decay", 0.9f);
    if (learning_rate_decay <= 0.0f || learning_rate_decay > 1.0f) {
        cerr << "decay wrong" << endl;
        abort();
    }
    hyper_params.learning_rate_decay = learning_rate_decay;

    float warm_up_learning_rate = ini_reader.GetReal("hyper", "warm_up_learning_rate", 1e-6);
    if (warm_up_learning_rate < 0 || warm_up_learning_rate > 1.0f) {
        cerr << "warm_up_learning_rate wrong" << endl;
        abort();
    }
    hyper_params.warm_up_learning_rate = warm_up_learning_rate;

    int warm_up_iterations = ini_reader.GetInteger("hyper", "warm_up_iterations", 1000);
    if (warm_up_iterations < 0) {
        cerr << "warm_up_iterations wrong" << endl;
        abort();
    }
    hyper_params.warm_up_iterations = warm_up_iterations;

    int word_cutoff = ini_reader.GetReal("hyper", "word_cutoff", -1);
    if(word_cutoff == -1){
   	cerr << "word_cutoff read error" << endl;
    }
    hyper_params.word_cutoff = word_cutoff;

    bool word_finetune = ini_reader.GetBoolean("hyper", "word_finetune", -1);
    hyper_params.word_finetune = word_finetune;

    string word_file = ini_reader.Get("hyper", "word_file", "");
    hyper_params.word_file = word_file;

    float l2_reg = ini_reader.GetReal("hyper", "l2_reg", 0.0f);
    if (l2_reg < 0.0f || l2_reg > 1.0f) {
        cerr << "l2_reg:" << l2_reg << endl;
        abort();
    }
    hyper_params.l2_reg = l2_reg;
    string optimizer = ini_reader.Get("hyper", "optimzer", "");
    if (optimizer == "adam") {
        hyper_params.optimizer = Optimizer::ADAM;
    } else if (optimizer == "adagrad") {
        hyper_params.optimizer = Optimizer::ADAGRAD;
    } else if (optimizer == "adamw") {
        hyper_params.optimizer = Optimizer::ADAMW;
    } else {
        cerr << "invalid optimzer:" << optimizer << endl;
        abort();
    }

    return hyper_params;
}

vector<int> toIds(const vector<string> &sentence, LookupTable<Param> &lookup_table) {
    vector<int> ids;
    for (const string &word : sentence) {
	int xid = lookup_table.getElemId(word);
        ids.push_back(xid);
    }
    return ids;
}

void printWordIds(const vector<int> &word_ids, const LookupTable<Param> &lookup_table) {
    for (int word_id : word_ids) {
        cout << lookup_table.elems.from_id(word_id) << " ";
    }
    cout << endl;
}

void analyze(const vector<int> &results, const vector<int> &answers, Metric &metric) {
    if (results.size() != answers.size()) {
        cerr << "results size is not equal to answers size" << endl;
        cerr << boost::format("results size:%1% answers size:%2%\n") % results.size() %
            answers.size();
        abort();
    }

    int size = results.size();
    for (int i = 0; i < size; ++i) {
        ++metric.overall_label_count;
        if (results.at(i) == answers.at(i)) {
            ++metric.correct_label_count;
        }
    }
}

string saveModel(const HyperParams &hyper_params, ModelParams &model_params,
        const string &filename_prefix, int epoch) {
    cout << "saving model file..." << endl;
    auto t = time(nullptr);
    auto tm = *localtime(&t);
    ostringstream oss;
    oss << put_time(&tm, "%d-%m-%Y-%H-%M-%S");
    string filename = filename_prefix + oss.str() + "-epoch" + to_string(epoch);
#if USE_GPU
    model_params.copyFromDeviceToHost();
#endif

    Json::Value root;
    root["hyper_params"] = hyper_params.toJson();
    root["model_params"] = model_params.toJson();
    Json::StreamWriterBuilder builder;
    builder["commentStyle"] = "None";
    builder["indentation"] = "";
    unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    ofstream out(filename);
    writer->write(root, &out);
    out.close();
    cout << format("model file %1% saved") % filename << endl;
    return filename;
}

shared_ptr<Json::Value> loadModel(const string &filename) {
    ifstream is(filename.c_str());
    shared_ptr<Json::Value> root(new Json::Value);
    if (is) {
        cout << "loading model..." << endl;
        stringstream sstr;
        sstr << is.rdbuf();
        string str = sstr.str();
        Json::CharReaderBuilder builder;
        auto reader = unique_ptr<Json::CharReader>(builder.newCharReader());
        string error;
        if (!reader->parse(str.c_str(), str.c_str() + str.size(), root.get(), &error)) {
            cerr << boost::format("parse json error:%1%") % error << endl;
            abort();
        }
        cout << "model loaded" << endl;
    } else {
        cerr << format("failed to open is, error when loading %1%") % filename << endl;
        abort();
    }

    return root;
}

void loadModel(const DefaultConfig &default_config, HyperParams &hyper_params,
        ModelParams &model_params,
        const Json::Value *root,
        const function<void(const DefaultConfig &default_config, const HyperParams &hyper_params,
            ModelParams &model_params, const Alphabet*)> &allocate_model_params) {
    hyper_params.fromJson((*root)["hyper_params"]);
    hyper_params.print();
    allocate_model_params(default_config, hyper_params, model_params, nullptr);
    model_params.fromJson((*root)["model_params"]);
#if USE_GPU
    model_params.copyFromHostToDevice();
#endif
}

StanceCategory getStanceCategory(const unordered_map<string, Stance> &stance_table, int post_id,
        int response_id) {
    const Stance &stance = stance_table.at(getKey(post_id, response_id));
    return static_cast<StanceCategory>(max_element(stance.begin(), stance.end()) - stance.begin());
}

float metricTestPosts(const HyperParams &hyper_params, ModelParams &model_params,
        const vector<PostAndResponses> &post_and_responses_vector,
        const vector<vector<string>> &post_sentences,
        const vector<vector<string>> &response_sentences,
        const unordered_map<string, Stance> &stance_table) {
    cout << "metricTestPosts begin" << endl;
    hyper_params.print();
    globalPoolEnabled() = false;

    int right_count = 0;
    int total_count = 0;

    int i = 0;
    Graph *graph = nullptr;
    vector<Node *> res_reps;
    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
        if (i % hyper_params.batch_size == 0) {
            res_reps.clear();
            if (graph != nullptr) {
                delete graph;
            }
            graph = new Graph;
            for (int j = 0; j < hyper_params.batch_size; ++j) {
                int res_id = post_and_responses_vector.at((i + j) %
                        post_and_responses_vector.size()).response_ids.at(0);
                const vector<string> &res = response_sentences.at(res_id);
                Node *rep = sentenceRep(*graph, res, hyper_params, model_params,
                        model_params.response_encoder_params, model_params.response_rep_params,
                        nullptr, false);
                res_reps.push_back(rep);
            }
            graph->compute();
        }

        int post_id = post_and_responses.post_id;
        const auto &post = post_sentences.at(post_id);
        std::array<Node *, 3> post_reps;
        for (int stance  = 0; stance < 3; ++stance) {
            StanceCategory s = static_cast<StanceCategory>(stance);
            Node *post_rep = sentenceRep(*graph, post, hyper_params, model_params,
                    model_params.left_to_right_encoder_params, model_params.post_rep_params, &s,
                    false);
            post_reps.at(stance) = post_rep;
        }
        graph->compute();

        auto res_reps_copy = res_reps;
        for (int res_id : post_and_responses.response_ids) {
            const vector<string> &res = response_sentences.at(res_id);
            Graph inner_graph;
            Node *rep = sentenceRep(inner_graph, res, hyper_params, model_params,
                    model_params.response_encoder_params, model_params.response_rep_params,
                    nullptr, false);
            inner_graph.compute();
            auto rep_cpu_v = rep->val().toCpu();
            Node *bucket = n3ldg_plus::bucket(*graph, rep_cpu_v);
            res_reps_copy.at(i % hyper_params.batch_size) = bucket;

            StanceCategory stance_category = getStanceCategory(stance_table,
                    post_and_responses.post_id, res_id);
            Node *stance_conditioned_post = post_reps.at(stance_category);
            auto probs = selectionProbs(*graph, {stance_conditioned_post}, res_reps_copy);
            graph->compute();
            int predicted_id = predict(probs).front();
            ++total_count;
            if (predicted_id == i % hyper_params.batch_size) {
                ++right_count;
            }
        }

        cout << boost::format("%1% acc:%2% right:%3% total:%4%") % i %
            ((float)right_count / total_count) % right_count % total_count << endl;

        ++i;
    }
    delete graph;

    globalPoolEnabled() = true;

    return (float)right_count / total_count;
}

void computeMeanAndStandardDeviation(const vector<float> &nums, float &mean, float &sd) {
    float sum = 0;
    for (float num : nums) {
        sum += num;
    }
    mean = sum / nums.size();
    if (nums.size() == 1) {
        sd = 0;
    } else {
        float variance = 0;
        for (float num : nums) {
            float x = num - mean;
            variance += x * x;
        }
        variance /= (nums.size() - 1);
        sd = sqrt(variance);
    }
}

map<StanceCategory, vector<int>> splitResponsesByStance(int post_id,
        const vector<int> &response_ids,
        const unordered_map<string, Stance> &stance_table) {
    map<StanceCategory, vector<int>> result;
    for (int response_id : response_ids) {
        StanceCategory stance = getStanceCategory(stance_table, post_id, response_id);
        auto it = result.find(stance);
        if (it == result.end()) {
            vector<int> ids = {response_id};
            result.insert(make_pair(stance, ids));
        } else {
            it->second.push_back(response_id);
        }
    }
    return result;
}

pair<unordered_set<int>, unordered_set<int>> PostAndResponseIds(
        const vector<PostAndResponses> &post_and_responses_vector) {
    unordered_set<int> post_ids, response_ids;
    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
        post_ids.insert(post_and_responses.post_id);
        for (int id : post_and_responses.response_ids) {
            response_ids.insert(id);
        }
    }
    return make_pair(post_ids, response_ids);
}

unordered_set<string> knownWords(const unordered_map<string, int> &word_counts, int word_cutoff) {
    unordered_set<string> word_set;
    for (auto it : word_counts) {
        if (it.second > word_cutoff) {
            word_set.insert(it.first);
        }
    }
    return word_set;
}

unordered_set<string> knownWords(const vector<string> &words) {
    unordered_set<string> word_set;
    for (const string& w : words) {
        word_set.insert(w);
    }
    return word_set;
}

template<typename T>
void preserveVector(vector<T> &vec, int count, int seed) {
    default_random_engine engine(seed);
    shuffle(vec.begin(), vec.end(), engine);
    vec.erase(vec.begin() + std::min<int>(count, vec.size()), vec.end());
}

unordered_map<string, float> calculateIdf(const vector<vector<string>> sentences) {
    cout << "sentences size:" << sentences.size() << endl;
    unordered_map<string, int> doc_counts;
    int i = 0;
    for (const vector<string> &sentence : sentences) {
        if (i++ % 10000 == 0) {
            cout << i << " ";
        }
        set<string> words;
        for (const string &word : sentence) {
            words.insert(word);
        }

        for (const string &word : words) {
            auto it = doc_counts.find(word);
            if (it == doc_counts.end()) {
                doc_counts.insert(make_pair(word, 1));
            } else {
                ++doc_counts.at(word);
            }
        }
    }
    cout << endl;

    unordered_map<string, float> result;
    for (const auto &it : doc_counts) {
        float idf = log(sentences.size() / static_cast<float>(it.second));
        if (idf < 0.0) {
            cerr << "idf:" << idf << endl;
            abort();
        }

        utf8_string utf8(it.first);
        result.insert(make_pair(it.first, idf));
    }

    return result;
}

int main(int argc, const char *argv[]) {
    cout << "dtype size:" << sizeof(dtype) << endl;

    Options options("single-turn-conversation", "single turn conversation");
    options.add_options()
        ("config", "config file name", cxxopts::value<string>());
    auto args = options.parse(argc, argv);

    string configfilename = args["config"].as<string>();
    INIReader ini_reader(configfilename);
    if (ini_reader.ParseError() < 0) {
        cerr << "parse ini failed" << endl;
        abort();
    }

    DefaultConfig &default_config = GetDefaultConfig();
    default_config = parseDefaultConfig(ini_reader);
    cout << "default_config:" << endl;
    default_config.print();
    globalPoolEnabled() = (default_config.program_mode == ProgramMode::TRAINING);

#if USE_GPU
    n3ldg_cuda::InitCuda(default_config.device_id, default_config.memory_in_gb);
#endif

    HyperParams hyper_params = parseHyperParams(ini_reader);
    cout << "hyper_params:" << endl;
    hyper_params.print();

    vector<PostAndResponses> train_post_and_responses = readPostAndResponsesVector(
            default_config.train_pair_file);
    preserveVector(train_post_and_responses, default_config.train_sample_count,
            default_config.seed);
    cout << "train_post_and_responses_vector size:" << train_post_and_responses.size()
        << endl;
    vector<PostAndResponses> dev_post_and_responses = readPostAndResponsesVector(
            default_config.dev_pair_file);
    preserveVector(dev_post_and_responses, default_config.dev_sample_count, default_config.seed);
    cout << "dev_post_and_responses_vector size:" << dev_post_and_responses.size()
        << endl;
    vector<PostAndResponses> test_post_and_responses = readPostAndResponsesVector(
            default_config.test_pair_file);
    preserveVector(test_post_and_responses, default_config.test_sample_count, default_config.seed);
    cout << "test_post_and_responses_vector size:" << test_post_and_responses.size()
        << endl;

    auto stance_table = readStanceTable(default_config.human_stance_file,
            default_config.auto_stance_file);

    vector<ConversationPair> train_conversation_pairs;
    for (const PostAndResponses &post_and_responses : train_post_and_responses) {
        vector<ConversationPair> conversation_pairs = toConversationPairs(post_and_responses,
                stance_table);
        for (ConversationPair &conversation_pair : conversation_pairs) {
            train_conversation_pairs.push_back(move(conversation_pair));
        }
    }

    cout << "train size:" << train_conversation_pairs.size() << " dev size:" <<
        dev_post_and_responses.size() << " test size:" << test_post_and_responses.size() << endl;

    vector<vector<string>> post_sentences = readSentences(default_config.post_file);
    vector<vector<string>> response_sentences = readSentences(default_config.response_file);

    vector<vector<string>> all_sentences;
    for (auto &p : train_conversation_pairs) {
        auto &s = response_sentences.at(p.response_id);
        all_sentences.push_back(s);
        auto &s2 = post_sentences.at(p.post_id);
        all_sentences.push_back(s2);
    }

    auto all_idf = calculateIdf(all_sentences);

    Alphabet alphabet;
    shared_ptr<Json::Value> root_ptr;
    unordered_map<string, int> word_counts;
    if (default_config.program_mode == ProgramMode::TRAINING) {
        auto wordStat = [&]() {
            for (const ConversationPair &conversation_pair : train_conversation_pairs) {
                const vector<string> &post_sentence = post_sentences.at(conversation_pair.post_id);
                addWord(word_counts, post_sentence);

                const vector<string> &response_sentence = response_sentences.at(
                        conversation_pair.response_id);
                addWord(word_counts, response_sentence);
            }

            if (hyper_params.word_file != "" && !hyper_params.word_finetune) {
                for (const PostAndResponses &dev : dev_post_and_responses){
                    const vector<string>&post_sentence = post_sentences.at(dev.post_id);
                    addWord(word_counts, post_sentence);

                    for(int i=0; i<dev.response_ids.size(); i++){
                        const vector<string>&resp_sentence = response_sentences.at(
                                dev.response_ids.at(i));
                        addWord(word_counts, resp_sentence);
                    }
                }

                for (const PostAndResponses &test : test_post_and_responses){
                    const vector<string>&post_sentence = post_sentences.at(test.post_id);
                    addWord(word_counts, post_sentence);

                    for(int i =0; i<test.response_ids.size(); i++){
                        const vector<string>&resp_sentence =
                            response_sentences.at(test.response_ids.at(i));
                        addWord(word_counts, resp_sentence);
                    }
                }
            }
        };
        wordStat();
        word_counts[unknownkey] = 1000000000;
        alphabet.init(word_counts, hyper_params.word_cutoff);
        cout << boost::format("post alphabet size:%1%") % alphabet.size() << endl;
    } else if (default_config.split_unknown_words) {
        root_ptr = loadModel(default_config.input_model_file);
        Json::Value &root = *root_ptr;
        vector<string> words = stringVectorFromJson(
                root["model_params"]["lookup_table"]["word_ids"]["m_id_to_string"]);
        unordered_set<string> word_set = knownWords(words);
        auto &v = default_config.program_mode == ProgramMode::METRIC ? dev_post_and_responses :
            test_post_and_responses;
        auto post_ids_and_response_ids = PostAndResponseIds(v);
        post_sentences = reprocessSentences(post_sentences, word_set,
                post_ids_and_response_ids.first);
        response_sentences = reprocessSentences(response_sentences, word_set,
                post_ids_and_response_ids.second);
    }

    ModelParams model_params;
    int beam_size = hyper_params.beam_size;

    auto allocate_model_params = [](const DefaultConfig &default_config,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            const Alphabet *alphabet) {
        cout << format("allocate word_file:%1%\n") % hyper_params.word_file;
        if (alphabet != nullptr) {
            if(hyper_params.word_file != "" &&
                    default_config.program_mode == ProgramMode::TRAINING &&
                    default_config.input_model_file == "") {
                model_params.lookup_table.init(*alphabet, hyper_params.word_file,
                        hyper_params.word_finetune);
            } else {
                model_params.lookup_table.init(*alphabet, hyper_params.word_dim, true);
            }
        }
        model_params.stance_embeddings.init(hyper_params.stance_dim, 3);
        model_params.left_to_right_encoder_params.init(hyper_params.hidden_dim,
                hyper_params.word_dim + hyper_params.stance_dim);
        model_params.response_encoder_params.init(hyper_params.hidden_dim, hyper_params.word_dim);
        model_params.response_rep_params.init(hyper_params.hidden_dim, hyper_params.hidden_dim);
        model_params.post_rep_params.init(hyper_params.hidden_dim, hyper_params.hidden_dim);
    };

    if (default_config.program_mode != ProgramMode::METRIC) {
        if (default_config.input_model_file == "") {
            allocate_model_params(default_config, hyper_params, model_params, &alphabet);
        } else {
            root_ptr = loadModel(default_config.input_model_file);
            loadModel(default_config, hyper_params, model_params, root_ptr.get(),
                    allocate_model_params);
            hyper_params.learning_rate_decay = ini_reader.GetFloat("hyper", "learning_rate_decay",
                    0);
            hyper_params.min_learning_rate = ini_reader.GetFloat("hyper", "min_learning_rate",
                    0);
            hyper_params.learning_rate = ini_reader.GetFloat("hyper", "learning_rate",
                    0);
            hyper_params.batch_size = ini_reader.GetFloat("hyper", "batch_size", 1);
            hyper_params.print();
        }
    }

    auto black_list = readBlackList(default_config.black_list_file);

    if (default_config.program_mode == ProgramMode::DECODING) {
        hyper_params.beam_size = beam_size;
        abort();
    } else if (default_config.program_mode == ProgramMode::METRIC) {
        path dir_path(default_config.input_model_dir);
        if (!is_directory(dir_path)) {
            cerr << format("%1% is not dir path") % default_config.input_model_dir << endl;
            abort();
        }

        vector<string> ordered_file_paths;
        for(auto& entry : boost::make_iterator_range(directory_iterator(dir_path), {})) {
            string basic_name = entry.path().filename().string();
            cout << format("basic_name:%1%") % basic_name << endl;
            if (basic_name.find("model") != 0) {
                continue;
            }

            string model_file_path = entry.path().string();
            ordered_file_paths.push_back(model_file_path);
        }
        std::sort(ordered_file_paths.begin(), ordered_file_paths.end(),
                [](const string &a, const string &b)->bool {
                using boost::filesystem::last_write_time;
                return last_write_time(a) < last_write_time(b);
                });

        float min_perplex = 0.0f;
        for(const string &model_file_path : ordered_file_paths) {
            cout << format("model_file_path:%1%") % model_file_path << endl;
            ModelParams model_params;
            shared_ptr<Json::Value> root_ptr = loadModel(model_file_path);
            loadModel(default_config, hyper_params, model_params, root_ptr.get(),
                    allocate_model_params);
            float perplex = metricTestPosts(hyper_params, model_params,
                    dev_post_and_responses, post_sentences, response_sentences, stance_table);
            cout << format("model %1% perplex is %2%") % model_file_path % perplex << endl;
            if (min_perplex > perplex) {
                min_perplex = perplex;
                cout << format("best model now is %1%, and perplex is %2%") % model_file_path %
                    perplex << endl;
            }
        }
    } else if (default_config.program_mode == ProgramMode::TRAINING) {
        ModelUpdate model_update;
        model_update._alpha = hyper_params.learning_rate;
        model_update._reg = hyper_params.l2_reg;
        model_update.setParams(model_params.tunableParams());

        CheckGrad grad_checker;
        if (default_config.check_grad) {
            grad_checker.init(model_params.tunableParams());
        }

        dtype loss_sum = 0.0f;
        dtype last_dev_acc = 0.0;

        string last_saved_model;

        for (int epoch = 0; epoch < default_config.max_epoch; ++epoch) {
            cout << "epoch:" << epoch << endl;

            model_params.lookup_table.E.is_fixed = false;

            auto cmp = [&] (const ConversationPair &a, const ConversationPair &b)->bool {
                auto len = [&] (const ConversationPair &pair)->int {
                    return post_sentences.at(pair.post_id).size() +
                        response_sentences.at(pair.response_id).size();
                };
                return len(a) < len(b);
            };
            sort(begin(train_conversation_pairs), end(train_conversation_pairs), cmp);
            int valid_len = train_conversation_pairs.size() / hyper_params.batch_size *
                hyper_params.batch_size;
            int batch_count = valid_len / hyper_params.batch_size;
            cout << boost::format("valid_len:%1% batch_count:%2%") % valid_len % batch_count <<
                endl;
            default_random_engine engine(default_config.seed);
            for (int i = 0; i < hyper_params.batch_size; ++i) {
                auto begin_pos = begin(train_conversation_pairs) + i * batch_count;
                shuffle(begin_pos, begin_pos + batch_count, engine);
            }
            if (train_conversation_pairs.size() > hyper_params.batch_size * batch_count) {
                shuffle(begin(train_conversation_pairs) + hyper_params.batch_size * batch_count,
                        train_conversation_pairs.end(), engine);
            }


            unique_ptr<Metric> metric = unique_ptr<Metric>(new Metric);
            using namespace std::chrono;

            for (int batch_i = 0; batch_i < batch_count +
                    (train_conversation_pairs.size() > hyper_params.batch_size * batch_count);
                    ++batch_i) {
                vector<int> answers;
                int batch_size = batch_i == batch_count ?
                    train_conversation_pairs.size() % hyper_params.batch_size :
                    hyper_params.batch_size;
                Graph graph;
                vector<ConversationPair> conversation_pair_in_batch;
                auto getSentenceIndex = [batch_i, batch_count](int i) {
                    return i * batch_count + batch_i;
                };
                vector<Node *> post_reps, res_reps;
                for (int i = 0; i < batch_size; ++i) {
                    int instance_index = getSentenceIndex(i);
                    int post_id = train_conversation_pairs.at(instance_index).post_id;
                    conversation_pair_in_batch.push_back(train_conversation_pairs.at(
                                instance_index));
                    int response_id = train_conversation_pairs.at(instance_index).response_id;
                    answers.push_back(i);
                    StanceCategory stance_category = getStanceCategory(stance_table, post_id,
                            response_id);
                    Node *post_rep = sentenceRep(graph, post_sentences.at(post_id), hyper_params,
                            model_params, model_params.left_to_right_encoder_params,
                            model_params.post_rep_params, &stance_category, true);
                    post_reps.push_back(post_rep);
                    Node *res_rep = sentenceRep(graph, response_sentences.at(response_id),
                            hyper_params, model_params, model_params.response_encoder_params,
                            model_params.response_rep_params, nullptr, true);
                    res_reps.push_back(res_rep);
                }
                graph.compute();
                auto probs = selectionProbs(graph, post_reps, res_reps);
                graph.compute();

                dtype loss = crossEntropyLoss(probs, answers, 1.0 / hyper_params.batch_size);
                vector<int> predicted_ids = predict(probs);
                graph.backward();
                model_update.updateAdam(5.0f);
                loss_sum += loss;
                analyze(predicted_ids, answers, *metric);
                if (batch_i % 100 == 0) {
                    cout << "batch_i:" << batch_i << endl;
                    cout << "avg loss:" << loss_sum / (batch_i + 1) << endl;;
                    metric->print();
                    int instance_index = getSentenceIndex(0);
                    int response_id = train_conversation_pairs.at(instance_index).response_id;

                    static int count_for_print;
                    count_for_print = 0;
                    int post_id = train_conversation_pairs.at(instance_index).post_id;
                    cout << "post:" << post_id << endl;
                    print(post_sentences.at(post_id));
                    const auto &stance =
                        train_conversation_pairs.at(instance_index).stance;
                    cout << boost::format("stance: %1%,%2%,%3%") % stance.at(0) %
                        stance.at(1) % stance.at(2) << endl;
                    cout << "golden answer:" << endl;
                    print(response_sentences.at(response_id));
                    cout << "output:" << endl;
                    int predicted_ins_index = getSentenceIndex(predicted_ids.at(0));
                    int predicted_res_id =
                        train_conversation_pairs.at(predicted_ins_index).response_id;
                    print(response_sentences.at(predicted_res_id));
                }
            }

            float dev_acc = metricTestPosts(hyper_params, model_params, dev_post_and_responses,
                    post_sentences, response_sentences, stance_table);
            cout << "dev acc:" << dev_acc << endl;

            float test_acc = metricTestPosts(hyper_params, model_params, test_post_and_responses,
                    post_sentences, response_sentences, stance_table);
            cout << "test acc:" << test_acc << endl;

            cout << "loss_sum:" << loss_sum << " last_loss_sum:" << endl;
            if (last_dev_acc > dev_acc) {
                if (epoch == 0) {
                    cerr << "loss is larger than last epoch but epoch is 0" << endl;
                    abort();
                }
                model_update._alpha *= 0.1f;
                hyper_params.learning_rate = model_update._alpha;
                cout << "learning_rate decay:" << model_update._alpha << endl;
            } else {
                model_update._alpha = (model_update._alpha - hyper_params.min_learning_rate) *
                    hyper_params.learning_rate_decay + hyper_params.min_learning_rate;
                hyper_params.learning_rate = model_update._alpha;
                cout << "learning_rate now:" << hyper_params.learning_rate << endl;
                last_saved_model = saveModel(hyper_params, model_params,
                        default_config.output_model_file_prefix, epoch);
            }

            last_dev_acc = dev_acc;
            loss_sum = 0;
        }
    } else {
        abort();
    }

    return 0;
}
