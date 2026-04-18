#pragma once
#include <vector>
#include <cstdint>

struct TwiddleTable {
    std::vector<uint64_t> forward;  // w^k, natural order, length N/2
    std::vector<uint64_t> inverse;  // w_inv^k, natural order, length N/2
    uint64_t n_inv;                 // N^{-1} mod q
};

TwiddleTable BuildTwiddleTable(uint64_t q, uint32_t N);
