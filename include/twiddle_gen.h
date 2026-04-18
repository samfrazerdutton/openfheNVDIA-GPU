#pragma once
#include <vector>
#include <cstdint>

// TwiddleTable for NEGACYCLIC NTT (ring Z[X]/(X^N + 1)).
// Requires a primitive 2N-th root psi: psi^(2N) = 1, psi^N = -1 mod q.
// forward[k] = psi^(bit_reverse(k)+1) * psi^k  (twisted NTT twiddles)
// inverse[k] = corresponding inverse twiddles
// n_inv       = N^{-1} mod q
struct TwiddleTable {
    std::vector<uint64_t> forward;  // length N (twisted forward twiddles)
    std::vector<uint64_t> inverse;  // length N (twisted inverse twiddles)
    uint64_t n_inv;                 // N^{-1} mod q
};

// Build negacyclic twiddle table for given NTT-friendly prime q and ring dim N.
// Requires (q - 1) % (2*N) == 0.
TwiddleTable BuildTwiddleTable(uint64_t q, uint32_t N);

