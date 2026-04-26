#pragma once
/**
 * Phase 4: GPU-Resident Hybrid Key-Switching
 *
 * Implements the RNS Hybrid Key-Switch:
 *   1. Digit decomposition of c1 into dnum groups
 *   2. NTT-forward of each digit (if in coeff domain)
 *   3. Dot product with evk (evaluation key) towers
 *   4. RNS sum reduction → new (c0', c1')
 *
 * Reference: Han & Ki, "Better Bootstrapping for Approximate
 * Homomorphic Encryption", CT-RSA 2020; and
 * Jung et al., ePrint 2021/508
 *
 * GPU memory model: evk is uploaded ONCE and stays resident
 * for the lifetime of the session (Phase 4 goal).
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <vector>

// Evaluation key tile (one digit's worth of evk towers)
struct EvkTile {
    uint64_t* d_evk_c0;   // device ptr — (dnum * L * N) uint64_t
    uint64_t* d_evk_c1;
    size_t    size_bytes;  // per pointer
};

// Parameters for one hybrid key-switch operation
struct KeySwitchParams {
    uint32_t  ring_dim;        // N (e.g. 32768)
    uint32_t  num_towers;      // L  (current ciphertext level towers)
    uint32_t  num_special;     // K  (number of special primes in evk)
    uint32_t  dnum;            // digit count = ceil((L+K)/alpha)
    uint32_t  alpha;           // towers per digit (typically 3-4)
    uint64_t* d_moduli;        // device array of L+K primes, length L+K
    uint64_t* d_barrett_mu;    // Barrett constants for each prime
};

class GpuKeySwitch {
public:
    static GpuKeySwitch& Instance();

    /**
     * Upload evaluation key once at key-gen time.
     * evk_c0, evk_c1: host arrays of size dnum*(L+K)*N uint64_t
     * Returns false if VRAM insufficient.
     */
    bool UploadEvk(const uint64_t* evk_c0,
                   const uint64_t* evk_c1,
                   const KeySwitchParams& params,
                   cudaStream_t stream);

    /**
     * Perform full hybrid key-switch on GPU.
     * d_c1_in:  input  c1 polynomial (device, L towers, N coefficients each)
     * d_c0_out: output c0 adjustment (device, L towers)
     * d_c1_out: output c1 adjustment (device, L towers)
     * All arrays must be pre-allocated on device.
     */
    void SwitchKey(const uint64_t* d_c1_in,
                   uint64_t* d_c0_out,
                   uint64_t* d_c1_out,
                   const KeySwitchParams& params,
                   cudaStream_t stream);

    bool IsEvkLoaded() const { return evk_loaded_; }
    void FreeEvk();

private:
    GpuKeySwitch() = default;
    uint64_t* d_evk_c0_ = nullptr;
    uint64_t* d_evk_c1_ = nullptr;
    uint64_t* d_moduli_  = nullptr;
    uint64_t* d_barrett_ = nullptr;
    size_t    evk_size_bytes_ = 0;
    bool      evk_loaded_     = false;
    KeySwitchParams params_{};
};

// CUDA kernel launchers (implemented in cuda_keyswitch.cu)
extern "C" {
void LaunchDigitDecompose(
    const uint64_t* d_c1,       // input: L towers × N
    uint64_t*       d_digits,   // output: dnum × (L+K) × N (zero-padded)
    const uint64_t* d_moduli,
    uint32_t N, uint32_t L, uint32_t K,
    uint32_t dnum, uint32_t alpha,
    cudaStream_t stream);

void LaunchEvkDotProduct(
    const uint64_t* d_digits,   // dnum × (L+K) × N
    const uint64_t* d_evk_c0,  // dnum × (L+K) × N
    const uint64_t* d_evk_c1,
    uint64_t*       d_out_c0,  // (L+K) × N  (will be mod-reduced after)
    uint64_t*       d_out_c1,
    const uint64_t* d_moduli,
    const uint64_t* d_barrett_mu,
    uint32_t N, uint32_t L, uint32_t K, uint32_t dnum,
    cudaStream_t stream);

void LaunchRnsModDown(
    const uint64_t* d_in,       // (L+K) × N
    uint64_t*       d_out,      // L × N  — special primes removed
    const uint64_t* d_moduli,
    uint32_t N, uint32_t L, uint32_t K,
    cudaStream_t stream);
}
