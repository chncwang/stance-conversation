#ifndef SINGLE_TURN_CONVERSATION_SRC_BASIC_DATA_MANAGER_H
#define SINGLE_TURN_CONVERSATION_SRC_BASIC_DATA_MANAGER_H

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <codecvt>
#include <fstream>
#include <iterator>
#include <regex>
#include <iostream>
#include <utility>
#include <atomic>
#include <mutex>
#include "single_turn_conversation/conversation_structure.h"
#include "single_turn_conversation/def.h"
#include "single_turn_conversation/default_config.h"
#include "tinyutf8.h"
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include <boost/algorithm/string/regex.hpp>
#include <boost/asio.hpp>

using namespace std;
using boost::format;
using namespace boost::asio;

vector<PostAndResponses> readPostAndResponsesVector(const string &filename) {
    vector<PostAndResponses> results;
    string line;
    ifstream ifs(filename);
    while (getline(ifs, line)) {
        vector<string> strs;
        boost::split(strs, line, boost::is_any_of(":"));
        if (strs.size() != 2) {
            abort();
        }
        int post_id = stoi(strs.at(0));
        PostAndResponses post_and_responses;
        post_and_responses.post_id = post_id;
        vector<string> strs2;
        boost::split(strs2, strs.at(1), boost::is_any_of(","));
        if (strs2.empty()) {
            cerr << "readPostAndResponsesVector - no response id found!" << line << endl;
            abort();
        }
        for (string &str : strs2) {
            post_and_responses.response_ids.push_back(stoi(str));
        }
        results.push_back(move(post_and_responses));
    }

    return results;
}

string getKey(int post_id, int response_id) {
    return to_string(post_id) + "-" + to_string(response_id);
}

vector<ConversationPair> toConversationPairs(const PostAndResponses &post_and_responses,
        const unordered_map<string, Stance> &stance_table) {
    vector<ConversationPair> results;
    for (int response_id : post_and_responses.response_ids) {
        string key = getKey(post_and_responses.post_id, response_id);
        const Stance &stance = stance_table.at(key);
        ConversationPair conversation_pair(post_and_responses.post_id, response_id, stance);
        results.push_back(move(conversation_pair));
    }
    return results;
}

vector<ConversationPair> toConversationPairs(
        const vector<PostAndResponses> &post_and_responses_vector,
        const unordered_map<string, Stance> &stance_table) {
    vector<ConversationPair> results;
    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
        vector<ConversationPair> conversation_pairs = toConversationPairs(post_and_responses,
                stance_table);
        for (const ConversationPair & conversation_pair : conversation_pairs) {
            results.push_back(conversation_pair);
        }
    }
    return results;
}

unordered_map<string, Stance> readHumanAnnotatedStanceTable(const string &filename) {
    unordered_map<string, Stance> result_table;
    string line;
    ifstream ifs(filename);
    while (getline(ifs, line)) {
        vector<string> strs;
        boost::split(strs, line, boost::is_any_of(" "));
        int post_id = stoi(strs.at(0));
        int response_id = stoi(strs.at(1));
        const string &stance = strs.at(2);
        Stance stance_dist;
        if (stance == "f") {
            stance_dist = {1, 0, 0};
        } else if (stance == "a") {
            stance_dist = {0, 1, 0};
        } else {
            stance_dist = {0, 0, 1};
        }
        string key = getKey(post_id, response_id);
        result_table.insert(make_pair(key, stance_dist));
    }
    return result_table;
}

unordered_map<string, Stance> readAutoAnnotatedStanceTable(const string &filename) {
    unordered_map<string, Stance> result_table;
    string line;
    ifstream ifs(filename);
    while (getline(ifs, line)) {
        vector<string> strs;
        boost::split(strs, line, boost::is_any_of(","));
        int post_id = stoi(strs.at(0));
        int response_id = stoi(strs.at(1));
        Stance stance_dist;
        for (int i = 0; i < 3; ++i) {
            stance_dist.at(i) = stof(strs.at(2 + i));
        }
        string key = getKey(post_id, response_id);
        result_table.insert(make_pair(key, stance_dist));
    }
    return result_table;
}

unordered_map<string, Stance> readStanceTable(const string &human_filename,
        const string &auto_filename) {
    auto human_table = readHumanAnnotatedStanceTable(human_filename);
    cout << "human_table size:" << human_table.size() << endl;
    auto auto_table = readAutoAnnotatedStanceTable(auto_filename);
    cout << "auto_table size:" << auto_table.size() << endl;
    for (const auto &it : auto_table) {
        const auto &human_it = human_table.find(it.first);
        if (human_it != human_table.end()) {
            auto_table.at(it.first) = human_it->second;
        }
    }
    return auto_table;
}

vector<ConversationPair> readConversationPairs(const string &filename,
        const string &stance_human_filename,
        const string &stance_auto_filename) {
    auto stance_table = readStanceTable(stance_human_filename, stance_auto_filename);
    vector<PostAndResponses> post_and_responses_vector = readPostAndResponsesVector(filename);
    vector<ConversationPair> results;
    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
        vector<ConversationPair> conversation_pairs = toConversationPairs(post_and_responses,
                stance_table);
        for (ConversationPair &conversation_pair : conversation_pairs) {
            results.push_back(move(conversation_pair));
        }
    }

    return results;
}

vector<vector<string>> readDecodedSentences(const string &filename) {
    vector<vector<string>> sentences;
    string line;
    ifstream ifs(filename);
    while (getline(ifs, line)) {
        vector<string> words;
        boost::split(words, line, boost::is_any_of(" "));
        sentences.push_back(move(words));
    }
    return sentences;
}

bool isPureChinese(const string &word) {
    regex expression("^[\u4e00-\u9fff]+$");
    return regex_search(word, expression);
}

bool containChinese(const utf8_string &word) {
    return word.size() != word.length();
}

bool isPureEnglish(const utf8_string &word) {
    if (containChinese(word)) {
        return false;
    }
    for (int i = 0; i < word.length(); ++i) {
        char c = word.at(i);
        if (!((c == '-' || c == '.' || c == '/' || c == ':' || c == '_') || (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))) {
            return false;
        }
    }
    return true;
}

bool isPureNumber(const utf8_string &word) {
    for (int i = 0; i < word.length(); ++i) {
        char c = word.at(i);
        if (!(c == '.' || (c >= '0' && c <= '9'))) {
            return false;
        }
    }
    return true;
}

vector<vector<string>> readSentences(const string &filename) {
    string line;
    ifstream ifs(filename);
    vector<vector<string>> results;

    int i = 0;
    while (getline(ifs, line)) {
        vector<string> strs;
        boost::split_regex(strs, line, boost::regex("##"));
        int index = stoi(strs.at(0));
        if (i != index) {
            abort();
        }

        const string &sentence = strs.at(1);
        vector<string> words;
        boost::split(words, sentence, boost::is_any_of(" "));
        vector<utf8_string> utf8_words;
        for (const string &word : words) {
            utf8_string s(word);
            utf8_words.push_back(s);
        }

        vector<string> characters;
        for (const utf8_string &word : utf8_words) {
            if (isPureEnglish(word) && !isPureNumber(word)) {
                string w;
                for (int i = 0; i < word.length(); ++i) {
                    char c = word.at(i);
                    if (c >= 'A' && c <= 'Z') {
                        c += 'a' - 'A';
                    }
                    w += c;
                }
                characters.push_back(w);
            } else {
                characters.push_back(word.cpp_str());
            }
        }

        characters.push_back(STOP_SYMBOL);
        results.push_back(characters);
        if (i % 10000 == 0) {
            cout << boost::format("i:%1%\n") % i;
            for (const string &c : characters) {
                cout << c << " ";
            }
            cout << endl;
        }
        ++i;
    }

    return results;
}

vector<string> reprocessSentence(const vector<string> &sentence,
        const unordered_map<string, int> &word_counts,
        int min_occurences) {
    vector<string> processed_sentence;
    for (const string &word : sentence) {
        if (isPureChinese(word)) {
            auto it = word_counts.find(word);
            int occurence;
            if (it == word_counts.end()) {
                cout << format("word not found:%1%\n") % word;
                occurence = 0;
            } else {
                occurence = it->second;
            }
            if (occurence <= min_occurences) {
                for (int i = 0; i < word.size(); i += 3) {
                    processed_sentence.push_back(word.substr(i, 3));
                }
            } else {
                processed_sentence.push_back(word);
            }
        } else {
            processed_sentence.push_back(word);
        }
    }
    return processed_sentence;
}

vector<vector<string>> reprocessSentences(const vector<vector<string>> &sentences,
        const unordered_set<string> &words,
        const unordered_set<int> &ids) {
    cout << boost::format("sentences size:%1%") % sentences.size() << endl;

    thread_pool pool(16);
    vector<vector<string>> result;
    map<int, vector<string>> result_map;
    mutex result_mutex;
    mutex cout_mutex;
    atomic_int i(0);
    int id = 0;
    for (const auto &sentence : sentences) {
        auto f = [&, id]() {
            if (i % 1000 == 0) {
                cout_mutex.lock();
                cout << static_cast<float>(i) / sentences.size() << endl;
                cout_mutex.unlock();
            }
            vector<string> processed_sentence;
            if (ids.find(id) == ids.end()) {
                processed_sentence = sentence;
            } else {
                for (const string &word : sentence) {
                    if (isPureChinese(word)) {
                        auto it = words.find(word);
                        if (it == words.end()) {
                            for (int i = 0; i < word.size(); i += 3) {
                                processed_sentence.push_back(word.substr(i, 3));
                            }
                        } else {
                            processed_sentence.push_back(word);
                        }
                    } else {
                        processed_sentence.push_back(word);
                    }
                }
            }
            result_mutex.lock();
            result_map.insert(make_pair(id, processed_sentence));
            result_mutex.unlock();
            ++i;
        };
        post(pool, f);
        ++id;
    }
    pool.join();

    for (int i = 0; i < sentences.size(); ++i) {
        auto it = result_map.find(i);
        if (it == result_map.end()) {
            cerr << boost::format("id %1% not found\n") % i;
            abort();
        }
        result.push_back(it->second);
    }

    return result;
}

void reprocessSentences(const vector<PostAndResponses> bundles,
        vector<vector<string>> &posts,
        vector<vector<string>> &responses,
        const unordered_map<string, int> &word_counts,
        int min_occurences) {
    vector<vector<string>> result;

    cout << "bund count:" << bundles.size() << endl;
    int i = 0;

    for (const PostAndResponses &bundle : bundles) {
        cout << i++ / (float)bundles.size() << endl;
        int post_id = bundle.post_id;
        auto &post = posts.at(post_id);
        post = reprocessSentence(post, word_counts, min_occurences);
        for (int response_id : bundle.response_ids) {
            auto &res = responses.at(response_id);
            res = reprocessSentence(res, word_counts, min_occurences);
        }
    }
}

vector<string> readBlackList(const string &filename) {
    string line;
    ifstream ifs(filename);
    vector<string> result;
    cout << "black:" << endl;
    while (getline(ifs, line)) {
        cout << line << endl;
        result.push_back(line);
    }
    return result;
}

#endif
