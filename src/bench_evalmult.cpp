#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <random>
using namespace std;
using Clock = chrono::high_resolution_clock;
extern "C" void gpu_rns_mult_wrapper(const uint64_t* a, const uint64_t* b, uint64_t* res,
    uint64_t q, uint64_t mu_hi, uint32_t ring, uint32_t tower_idx);
extern "C" void gpu_synchronize_all();
extern "C" void gpu_clear_vram_cache();
static bool is_prime(uint64_t n) {
    if (n<2) return false; if (n==2) return true; if (n%2==0) return false;
    for (uint64_t i=3;i*i<=n;i+=2) if(n%i==0) return false; return true;
}
static vector<uint64_t> gen_primes(uint32_t ring, int count) {
    vector<uint64_t> out; uint64_t step=2ULL*ring, c=(1ULL<<56)/step;
    while((int)out.size()<count){uint64_t q=c*step+1;if(is_prime(q))out.push_back(q);++c;}
    return out;
}
double benchmark_gpu(uint32_t towers, uint32_t ring, int iters) {
    mt19937_64 rng(42);
    auto moduli = gen_primes(ring, towers);
    vector<uint64_t> mu_hi(towers);
    vector<vector<uint64_t>> ha(towers,vector<uint64_t>(ring)),
                              hb(towers,vector<uint64_t>(ring)),
                              hres(towers,vector<uint64_t>(ring));
    for (uint32_t i=0;i<towers;++i) {
        mu_hi[i]=(uint64_t)(((unsigned __int128)1<<64)/moduli[i]);
        for (uint32_t j=0;j<ring;++j){ha[i][j]=rng()%moduli[i];hb[i][j]=rng()%moduli[i];}
    }
    for (uint32_t i=0;i<towers;++i)
        gpu_rns_mult_wrapper(ha[i].data(),hb[i].data(),hres[i].data(),moduli[i],mu_hi[i],ring,i);
    gpu_synchronize_all();
    gpu_clear_vram_cache();
    auto t0=Clock::now();
    for (int it=0;it<iters;++it){
        for (uint32_t i=0;i<towers;++i)
            gpu_rns_mult_wrapper(ha[i].data(),hb[i].data(),hres[i].data(),moduli[i],mu_hi[i],ring,i);
        gpu_synchronize_all();
        gpu_clear_vram_cache();
    }
    return chrono::duration<double,milli>(Clock::now()-t0).count()/iters;
}
int main() {
    printf("========================================\n CUDA HAL Benchmark\n========================================\n");
    struct TC{uint32_t towers,ring;const char* label;};
    TC cases[]={{16,16384,"16 towers, N=16k"},{16,32768,"16 towers, N=32k"},{16,65536,"16 towers, N=64k"}};
    for (auto& tc:cases) printf("  %-25s  GPU: %6.2f ms\n",tc.label,benchmark_gpu(tc.towers,tc.ring,20));
    return 0;
}
