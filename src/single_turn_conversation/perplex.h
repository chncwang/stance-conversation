#ifndef SINGLE_TURN_CONVERSATION_PERPLEX_H
#define SINGLE_TURN_CONVERSATION_PERPLEX_H

#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include <cmath>
#include <iostream>
#include <boost/format.hpp>

#include "n3ldg-plus/n3ldg-plus.h"

float computePerplex(const Node &node, int row, const std::vector<int> &answers, int &hit_count,
        vector<int> &hit_flags,
        int hit_beam) {
    if (hit_beam < 1) {
        cerr << "hit beam is less than 0" << hit_beam << endl;
        abort();
    }
    float log_sum = 0.0f;
    float count_sum = 0;
    hit_flags.clear();

    int col = node.getDim() / row;
    if (col * row != node.getDim()) {
        cerr << boost::format("computePerplex col:%1% node dim:%2%\n") % col % node.getDim()
            << endl;
    }
    for (int i = 0; i < col; ++i) {
        int answer = answers.at(i);
        if (answer < 0 || answer >= node.getDim()) {
            cerr << boost::format("answer:%1% dim:%2%") << answer << node.getDim() << endl;
            abort();
        }
#if USE_GPU
        const_cast<Node &>(node).val().copyFromDeviceToHost();
#endif
        float reciprocal_answer_prob = 1 / node.getVal()[row * i + answer];
        log_sum += log(reciprocal_answer_prob);

        bool hit = true;
        int larger_count = 0;
        for (int j = 0; j < row; ++j) {
            if (node.getVal()[row * i + j] >= node.getVal()[row * i + answer]) {
                if (++larger_count > hit_beam) {
                    hit = false;
                    break;
                }
            }
        }
        if (hit) {
            ++count_sum;
        }
        hit_flags.push_back(hit);
    }

    hit_count = count_sum;
    return log_sum;
}

#endif
