#ifndef SINGLE_TURN_CONVERSATION_SRC_BASIC_CONVERSATION_STRUCTURE_H
#define SINGLE_TURN_CONVERSATION_SRC_BASIC_CONVERSATION_STRUCTURE_H

#include <vector>
#include <array>

struct PostAndResponses {
    int post_id;
    std::vector<int> response_ids;
};

typedef std::array<float, 3> Stance;

enum StanceCategory {
    FAVOR = 0,
    AGAINST = 1,
    NEUTRAL = 2
};

struct ConversationPair {
    ConversationPair() = default;

    ConversationPair(int postid, int responseid, const Stance &stanc) : post_id(postid),
    response_id(responseid), stance(stanc) {}

    int post_id;
    int response_id;
    Stance stance;
};

#endif
