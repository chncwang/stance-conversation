#ifndef SINGLE_TURN_CONVERSATION_DEFAULT_CONFIG_H
#define SINGLE_TURN_CONVERSATION_DEFAULT_CONFIG_H

#include <iostream>
#include <string>

using namespace std;

enum ProgramMode {
    TRAINING = 0,
    DECODING = 1,
    INTERACTING = 2,
    METRIC = 3,
    DECODED_PPL = 4
};

struct DefaultConfig {
    string train_pair_file;
    string dev_pair_file;
    string test_pair_file;
    string post_file;
    string response_file;
    string decoded_file;
    ProgramMode program_mode;
    bool check_grad;
    bool one_response;
    bool learn_test;
    bool save_model_per_batch;
    bool split_unknown_words;
    int train_sample_count;
    int dev_sample_count;
    int test_sample_count;
    int device_id;
    int hold_batch_size;
    int seed;
    int cut_length;
    int max_epoch;
    string output_model_file_prefix;
    string input_model_file;
    string input_model_dir;
    string black_list_file;
    float memory_in_gb;
    float result_count_factor;
    string human_stance_file;
    string auto_stance_file;

    void print() const {
        cout << "train_pair_file:" << train_pair_file << endl
            << "dev_pair_file:" << dev_pair_file << endl
            << "test_pair_file:" << test_pair_file << endl
            << "decoded_file:" << decoded_file << endl
            << "post_file:" << post_file << endl
            << "response_file:" << response_file << endl
            << "program_mode:" << program_mode << endl
            << "check_grad:" << check_grad << endl
            << "one_response:" << one_response << endl
            << "learn_test:" << learn_test << endl
            << "save_model_per_batch:" << save_model_per_batch << endl
            << "split_unknown_words:" << split_unknown_words << endl
            << "train_sample_count:" << train_sample_count << endl
            << "dev_sample_count:" << dev_sample_count << endl
            << "test_sample_count:" << test_sample_count << endl
            << "device_id:" << device_id << endl
            << "hold_batch_size:" << hold_batch_size << endl
            << "seed:" << seed << endl
            << "cut_length:" << cut_length << endl
            << "max_epoch:" << max_epoch << endl
            << "output_model_file_prefix" << output_model_file_prefix << endl
            << "input_model_file:" << input_model_file << endl
            << "input_model_dir:" << input_model_dir << endl
            << "black_list_file:" << black_list_file << endl
            << "memory_in_gb:" << memory_in_gb << endl
            << "human_stance_file:" << human_stance_file << endl
            << "auto_stance_file:" << auto_stance_file << endl
            << "result_count_factor:" << result_count_factor << endl;
    }
};

DefaultConfig &GetDefaultConfig() {
    static DefaultConfig default_config;
    return default_config;
}

#endif
