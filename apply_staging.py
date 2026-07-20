#!/usr/bin/env python3
"""Apply pinned-staging to gpu_rns_mult_batch_wrapper in src/cuda_hal.cpp.

Data path becomes: host --memcpy--> pinned staging --async DMA--> device
                   device --async DMA--> pinned staging --memcpy--> host

Safe by construction: the HAL owns the pinned memory (cudaMallocHost),
so OpenFHE freeing its own buffers can never dangle a registration.
Falls back to pageable copies if pinned allocation fails.

Run from the repo root:  python3 apply_staging.py
"""
import sys, os

P = "src/cuda_hal.cpp"
if not os.path.exists(P):
    sys.exit("run from the repo root (src/cuda_hal.cpp not found)")

src = open(P).read()

if "PinnedStagingPool" in src:
    sys.exit("[=] already applied — nothing to do")

OLD = """    // d_out is now cached the same way da/db already were: keyed by the real,
    // unique output host pointer via ShadowRegistry. ShadowRegistry already
    // proved safe under 8 concurrent OMP threads for da/db -- reusing that
    // exact mechanism (rather than a new pool keyed only by tower index)
    // avoids the per-call cudaMalloc/cudaFree without introducing a new
    // buffer-aliasing race across concurrent callers.
    std::vector<uint64_t*> d_out(num_towers, nullptr);

    for (uint32_t i = 0; i < num_towers; i++) {
        cudaStream_t s = openfhe_cuda::StreamPool::Instance().Get(i);
        // NOTE: no cudaHostRegister here. OpenFHE frees temporaries while
        // still registered, leaving dangling pin state on reused addresses
        // (-> cudaMemcpyAsync invalid argument). Pinning needs HAL-owned
        // staging buffers; until then, pageable copies are correct and safe.
        uint64_t* da = reg.GetDevicePtr(ha[i], bytes);
        uint64_t* db = reg.GetDevicePtr(hb[i], bytes);
        d_out[i]      = reg.GetDevicePtr(hr[i], bytes);
        CUDA_CHECK(cudaMemcpyAsync(da, ha[i], bytes, cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(db, hb[i], bytes, cudaMemcpyHostToDevice, s));
        LaunchRNSMultMontgomery(da, db, d_out[i],
            q[i], calc_q_inv(q[i]), calc_R2(q[i]), ring, s);
        CUDA_CHECK(cudaMemcpyAsync(hr[i], d_out[i], bytes, cudaMemcpyDeviceToHost, s));
    }
    openfhe_cuda::StreamPool::Instance().SyncAll();"""

NEW = """    // Pinned staging path (Fix 1): host -> HAL-owned pinned buffer (memcpy)
    // -> device (true async DMA); results come back the same way. The HAL
    // owns the pinned memory (cudaMallocHost, reused forever), so OpenFHE
    // freeing its own buffers can never dangle a registration. If pinned
    // allocation fails we fall back to pageable copies (slower, correct).
    std::vector<uint64_t*> d_out(num_towers, nullptr);
    auto& pool = PinnedStagingPool::Instance();
    std::vector<void*> st_a(num_towers, nullptr), st_b(num_towers, nullptr),
                       st_r(num_towers, nullptr);
    bool staged = true;
    for (uint32_t i = 0; i < num_towers; i++) {
        st_a[i] = pool.Acquire(bytes);
        st_b[i] = pool.Acquire(bytes);
        st_r[i] = pool.Acquire(bytes);
        if (!st_a[i] || !st_b[i] || !st_r[i]) staged = false;
    }

    for (uint32_t i = 0; i < num_towers; i++) {
        cudaStream_t s = openfhe_cuda::StreamPool::Instance().Get(i);
        uint64_t* da = reg.GetDevicePtr(ha[i], bytes);
        uint64_t* db = reg.GetDevicePtr(hb[i], bytes);
        d_out[i]      = reg.GetDevicePtr(hr[i], bytes);
        if (staged) {
            std::memcpy(st_a[i], ha[i], bytes);
            std::memcpy(st_b[i], hb[i], bytes);
            CUDA_CHECK(cudaMemcpyAsync(da, st_a[i], bytes, cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(db, st_b[i], bytes, cudaMemcpyHostToDevice, s));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(da, ha[i], bytes, cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(db, hb[i], bytes, cudaMemcpyHostToDevice, s));
        }
        LaunchRNSMultMontgomery(da, db, d_out[i],
            q[i], calc_q_inv(q[i]), calc_R2(q[i]), ring, s);
        if (staged)
            CUDA_CHECK(cudaMemcpyAsync(st_r[i], d_out[i], bytes, cudaMemcpyDeviceToHost, s));
        else
            CUDA_CHECK(cudaMemcpyAsync(hr[i], d_out[i], bytes, cudaMemcpyDeviceToHost, s));
    }
    openfhe_cuda::StreamPool::Instance().SyncAll();

    for (uint32_t i = 0; i < num_towers; i++) {
        if (staged) std::memcpy(hr[i], st_r[i], bytes);
        pool.Release(st_a[i]);
        pool.Release(st_b[i]);
        pool.Release(st_r[i]);
    }"""

if OLD not in src:
    print("[!] ANCHOR NOT FOUND. Current wrapper region for diagnosis:")
    k = src.find("gpu_rns_mult_batch_wrapper")
    print(src[k:k + 2400] if k >= 0 else "(function name not found at all)")
    sys.exit(1)

src = src.replace(OLD, NEW, 1)

if "#include <cstring>" not in src:
    src = src.replace("#include <cuda_runtime.h>",
                      "#include <cuda_runtime.h>\n#include <cstring>", 1)

open(P, "w").write(src)
print("[+] pinned staging applied to gpu_rns_mult_batch_wrapper")
