#include "twiddle_gen.h"
#include <stdexcept>
#include <string>
#include <vector>

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

static void check_negacyclic_friendly(uint64_t q, uint32_t N) {
    if ((q - 1) % (2ULL * N) != 0)
        throw std::runtime_error("[twiddle_gen] q is not negacyclic-NTT-friendly for N");
}

static uint64_t find_generator(uint64_t q) {
    uint64_t phi = q - 1;
    for (uint64_t g = 2; g < q; g++) {
        bool ok = true;
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
    throw std::runtime_error("[twiddle_gen] No generator found");
}

TwiddleTable BuildTwiddleTable(uint64_t q, uint32_t N) {
    check_negacyclic_friendly(q, N);

    uint64_t g       = find_generator(q);
    uint64_t psi     = powmod(g, (q - 1) / (2ULL * N), q);
    uint64_t psi_inv = powmod(psi, q - 2, q);
    uint64_t w       = powmod(psi, 2, q);
    uint64_t w_inv   = powmod(w, q - 2, q);
    uint64_t n_inv   = powmod((uint64_t)N, q - 2, q);

    TwiddleTable tt;
    tt.n_inv = n_inv;
    
    // Allocate 2N: [0, N-1] for twist, [N, 1.5N-1] for cyclic DIT/DIF roots
    tt.forward.resize(2 * N, 0);
    tt.inverse.resize(2 * N, 0);

    // 1. Pre/Post Twist Roots (psi^k)
    uint64_t pk = 1, pkinv = 1;
    for (uint32_t k = 0; k < N; k++) {
        tt.forward[k] = pk;
        tt.inverse[k] = pkinv;
        pk    = (uint64_t)(((unsigned __int128)pk * psi) % q);
        pkinv = (uint64_t)(((unsigned __int128)pkinv * psi_inv) % q);
    }

    // 2. Standard Cyclic Roots (w^k)
    uint32_t N_half = N / 2;
    uint64_t wk = 1, wkinv = 1;
    for (uint32_t k = 0; k < N_half; k++) {
        tt.forward[N + k] = wk;
        tt.inverse[N + k] = wkinv;
        wk    = (uint64_t)(((unsigned __int128)wk * w) % q);
        wkinv = (uint64_t)(((unsigned __int128)wkinv * w_inv) % q);
    }

    return tt;
}
