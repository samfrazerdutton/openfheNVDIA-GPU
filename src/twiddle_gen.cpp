#include "twiddle_gen.h"
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>

static uint64_t powmod(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1; base %= mod;
    while (exp > 0) {
        if (exp & 1) result = (uint64_t)(((unsigned __int128)result * base) % mod);
        base = (uint64_t)(((unsigned __int128)base * base) % mod);
        exp >>= 1;
    }
    return result;
}

static void check_negacyclic_friendly(uint64_t q, uint32_t N) {
    if ((q - 1) % (2ULL * N) != 0)
        throw std::runtime_error(
            "[twiddle_gen] q=" + std::to_string(q) +
            " not negacyclic-NTT-friendly for N=" + std::to_string(N));
}

static uint64_t find_generator(uint64_t q) {
    uint64_t phi = q - 1;
    for (uint64_t g = 2; g < q; g++) {
        bool ok = true;
        uint64_t tmp = phi;
        for (uint64_t p = 2; p * p <= tmp; p++) {
            if (tmp % p == 0) {
                if (powmod(g, phi/p, q) == 1) { ok = false; break; }
                while (tmp % p == 0) tmp /= p;
            }
        }
        if (ok && tmp > 1 && powmod(g, phi/tmp, q) == 1) ok = false;
        if (ok) return g;
    }
    throw std::runtime_error("[twiddle_gen] No generator found");
}

// Layout: 2N entries.
//   [0..N-1]   = psi^0, psi^1, ..., psi^(N-1)        pre-twist
//   [N..2N-1]  = w^0,   w^1,   ..., w^(N-1)           cyclic NTT (w = psi^2)
TwiddleTable BuildTwiddleTable(uint64_t q, uint32_t N) {
    check_negacyclic_friendly(q, N);
    uint64_t g       = find_generator(q);
    uint64_t psi     = powmod(g, (q-1)/(2ULL*N), q);
    uint64_t psi_inv = powmod(psi, q-2, q);
    uint64_t w       = powmod(psi, 2, q);       // w = psi^2, N-th root of unity
    uint64_t w_inv   = powmod(w, q-2, q);
    uint64_t n_inv   = powmod((uint64_t)N, q-2, q);

    if (powmod(psi, 2ULL*N, q) != 1)
        throw std::runtime_error("[twiddle_gen] psi^(2N) != 1");
    if (powmod(psi, N, q) != q-1)
        throw std::runtime_error("[twiddle_gen] psi^N != -1");

    TwiddleTable tt;
    tt.n_inv = n_inv;
    tt.forward.resize(2*N);
    tt.inverse.resize(2*N);

    // Pre-twist section: psi^k and psi_inv^k for k=0..N-1
    uint64_t pk = 1, pk_inv = 1;
    for (uint32_t k = 0; k < N; k++) {
        tt.forward[k] = pk;
        tt.inverse[k] = pk_inv;
        pk     = (uint64_t)(((unsigned __int128)pk     * psi)     % q);
        pk_inv = (uint64_t)(((unsigned __int128)pk_inv * psi_inv) % q);
    }

    // Cyclic NTT section: w^j and w_inv^j for j=0..N-1
    uint64_t wj = 1, wj_inv = 1;
    for (uint32_t j = 0; j < N; j++) {
        tt.forward[N + j] = wj;
        tt.inverse[N + j] = wj_inv;
        wj     = (uint64_t)(((unsigned __int128)wj     * w)     % q);
        wj_inv = (uint64_t)(((unsigned __int128)wj_inv * w_inv) % q);
    }

    return tt;
}
