#pragma once
#include <vector>
#include <cstdint>

struct TwiddleTable {
    std::vector<uint64_t> forward;   // 2N layout: twist[0..N) ++ cyclic_roots[N..2N)
    std::vector<uint64_t> inverse;   // 2N layout: twist_inv[0..N) ++ cyclic_inv[N..2N)
    std::vector<uint32_t> bit_rev;   // precomputed bit-reversal permutation, length N
    uint64_t n_inv;                  // N^{-1} mod q
};

TwiddleTable BuildTwiddleTable(uint64_t q, uint32_t N);
