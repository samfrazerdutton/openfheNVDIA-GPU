#pragma once
#include <vector>
#include <cstdint>

struct TwiddleTable {
    std::vector<uint64_t> forward;  // twisted twiddles, length N
    std::vector<uint64_t> inverse;  // twisted inverse twiddles, length N
    uint64_t n_inv;                 // N^{-1} mod q
};

TwiddleTable BuildTwiddleTable(uint64_t q, uint32_t N);
