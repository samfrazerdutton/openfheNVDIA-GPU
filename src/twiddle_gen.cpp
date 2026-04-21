#include "twiddle_gen.h"
#include <stdexcept>
#include <string>
#include <vector>

static uint64_t powmod(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1; base %= mod;
    while (exp > 0) {
        if (exp & 1) result = (uint64_t)(((unsigned __int128)result * base) % mod);
        base = (uint64_t)(((unsigned __int128)base * base) % mod);
        exp >>= 1;
    }
    return result;
}

// Require (q-1) % 2N == 0 for negacyclic-NTT-friendly prime.
static void check_negacyclic_friendly(uint64_t q, uint32_t N) {
    if ((q - 1) % (2ULL * N) != 0)
        throw std::runtime_error(
            "[twiddle_gen] q=" + std::to_string(q) +
            " is not negacyclic-NTT-friendly for N=" + std::to_string(N) +
            " (need (q-1) % " + std::to_string(2ULL * N) + " == 0)");
}

static uint64_t find_generator(uint64_t q) {
    uint64_t phi = q - 1;
    for (uint64_t g = 2; g < q; g++) {
        bool ok = true; uint64_t tmp = phi;
        for (uint64_t p = 2; p * p <= tmp; p++) {
            if (tmp % p == 0) {
                if (powmod(g, phi / p, q) == 1) { ok = false; break; }
                while (tmp % p == 0) tmp /= p;
            }
        }
        if (ok && tmp > 1 && powmod(g, phi / tmp, q) == 1) ok = false;
        if (ok) return g;
    }
    throw std::runtime_error("[twiddle_gen] No primitive root found for q=" +
                             std::to_string(q));
}

// Build a 2N twiddle table for negacyclic NTT.
//
// Layout:
//   forward[0..N-1]     = psi^k mod q          (pre-twist roots)
//   forward[N..3N/2-1]  = w^k mod q, k=0..N/2-1 (cyclic DIT roots, w=psi^2)
//   inverse[0..N-1]     = psi_inv^k mod q       (post-untwist roots)
//   inverse[N..3N/2-1]  = w_inv^k mod q         (cyclic DIF roots)
//
// psi   = g^{(q-1)/(2N)} is a primitive 2N-th root: psi^N = -1 mod q.
// w     = psi^2           is a primitive N-th root.
// n_inv = N^{-1} mod q    for INTT scaling.
TwiddleTable BuildTwiddleTable(uint64_t q, uint32_t N) {
    check_negacyclic_friendly(q, N);

    const uint64_t g       = find_generator(q);
    const uint64_t psi     = powmod(g, (q - 1) / (2ULL * N), q);
    const uint64_t psi_inv = powmod(psi, q - 2, q);
    const uint64_t w       = powmod(psi, 2, q);   // psi^2
    const uint64_t w_inv   = powmod(w,   q - 2, q);
    const uint64_t n_inv   = powmod((uint64_t)N, q - 2, q);

    // Sanity checks.
    if (powmod(psi, N, q) != q - 1)
        throw std::runtime_error("[twiddle_gen] psi^N != -1 mod q");
    if (powmod(w, N, q) != 1)
        throw std::runtime_error("[twiddle_gen] w^N != 1 mod q");

    TwiddleTable tt;
    tt.n_inv = n_inv;

    const uint32_t N_half = N / 2;
    tt.forward.resize(2 * N, 0);
    tt.inverse.resize(2 * N, 0);

    // Section 1: twist roots psi^k and psi_inv^k for k in [0, N).
    uint64_t pk = 1, pkinv = 1;
    for (uint32_t k = 0; k < N; k++) {
        tt.forward[k] = pk;
        tt.inverse[k] = pkinv;
        pk    = (uint64_t)(((unsigned __int128)pk    * psi)     % q);
        pkinv = (uint64_t)(((unsigned __int128)pkinv * psi_inv) % q);
    }

    // Section 2: cyclic roots w^k and w_inv^k for k in [0, N/2).
    // Stored at offset N; the NTT kernel accesses tw + N as its root table.
    uint64_t wk = 1, wkinv = 1;
    for (uint32_t k = 0; k < N_half; k++) {
        tt.forward[N + k] = wk;
        tt.inverse[N + k] = wkinv;
        wk    = (uint64_t)(((unsigned __int128)wk    * w)     % q);
        wkinv = (uint64_t)(((unsigned __int128)wkinv * w_inv) % q);
    }

    return tt;
}
