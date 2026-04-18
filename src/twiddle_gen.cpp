#include "twiddle_gen.h"
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>

static uint64_t powmod(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) result = (uint64_t)(((unsigned __int128)result * base) % mod);
        base = (uint64_t)(((unsigned __int128)base * base) % mod);
        exp >>= 1;
    }
    return result;
}

// Check q is cyclic-NTT friendly: (q-1) % N == 0
static void check_ntt_friendly(uint64_t q, uint32_t N) {
    if ((q - 1) % N != 0)
        throw std::runtime_error(
            "[twiddle_gen] q=" + std::to_string(q) +
            " not NTT-friendly for N=" + std::to_string(N) +
            ". Need (q-1) % N == 0.");
}

// Find primitive root g of Z_q* (q prime)
static uint64_t find_generator(uint64_t q) {
    uint64_t phi = q - 1;
    for (uint64_t g = 2; g < q; g++) {
        bool ok  = true;
        uint64_t tmp = phi;
        for (uint64_t p = 2; p * p <= tmp; p++) {
            if (tmp % p == 0) {
                if (powmod(g, phi / p, q) == 1) { ok = false; break; }
                while (tmp % p == 0) tmp /= p;
            }
        }
        if (ok && tmp > 1 && powmod(g, phi / tmp, q) == 1) ok = false;
        if (ok) return g;
    }
    throw std::runtime_error("[twiddle_gen] No generator found for q=" + std::to_string(q));
}

// Builds twiddle tables for CYCLIC NTT (polynomial ring mod X^N - 1).
// w  = primitive N-th root of unity:  w^N = 1, w^(N/2) = -1 ... not needed,
//      just w = g^((q-1)/N).
// tw.forward[k] = w^k,     k in [0, N/2)
// tw.inverse[k] = w_inv^k, k in [0, N/2)
// tw.n_inv      = N^{-1} mod q
TwiddleTable BuildTwiddleTable(uint64_t q, uint32_t N) {
    check_ntt_friendly(q, N);
    uint64_t g    = find_generator(q);
    // Primitive N-th root (NOT 2N-th): w^N = 1
    uint64_t w    = powmod(g, (q - 1) / N, q);
    uint64_t winv = powmod(w, q - 2, q);      // w^{-1} mod q
    uint64_t ninv = powmod((uint64_t)N, q - 2, q);

    // Sanity checks
    if (powmod(w, N, q) != 1)
        throw std::runtime_error("[twiddle_gen] w^N != 1");
    if (powmod(w, N / 2, q) == 1)
        throw std::runtime_error("[twiddle_gen] w is not primitive (w^(N/2)==1)");

    TwiddleTable tt;
    tt.n_inv = ninv;
    tt.forward.resize(N / 2);
    tt.inverse.resize(N / 2);

    uint64_t wk = 1, wk_inv = 1;
    for (uint32_t k = 0; k < N / 2; k++) {
        tt.forward[k] = wk;
        tt.inverse[k] = wk_inv;
        wk     = (uint64_t)(((unsigned __int128)wk     * w)    % q);
        wk_inv = (uint64_t)(((unsigned __int128)wk_inv * winv) % q);
    }
    return tt;
}
