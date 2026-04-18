#include "cuda_hal.h"
#include <omp.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

using namespace std;
using clk = chrono::high_resolution_clock;

extern "C" void gpu_rns_mult_batch_wrapper(
    const uint64_t** ha, const uint64_t** hb, uint64_t** hr,
    const uint64_t* q, uint32_t ring, uint32_t num_towers);

extern "C" void gpu_poly_mult_wrapper(
    const uint64_t** ha, const uint64_t** hb, uint64_t** hr,
    const uint64_t* q, uint32_t ring, uint32_t num_towers);

static bool is_prime(uint64_t n) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0) return false;
    uint64_t d = n-1; int r = 0;
    while (d%2==0){d/=2;r++;}
    for (uint64_t a:{2ULL,3ULL,5ULL,7ULL,11ULL,13ULL,17ULL,19ULL,23ULL,29ULL,31ULL,37ULL}) {
        if (a>=n) continue;
        uint64_t x=1,b2=a%n,e2=d;
        while(e2>0){if(e2&1)x=(uint64_t)(((unsigned __int128)x*b2)%n);b2=(uint64_t)(((unsigned __int128)b2*b2)%n);e2>>=1;}
        if(x==1||x==n-1)continue;
        bool comp=true;
        for(int i=0;i<r-1;i++){x=(uint64_t)(((unsigned __int128)x*x)%n);if(x==n-1){comp=false;break;}}
        if(comp)return false;
    }
    return true;
}

// FIX: generate primes satisfying (q-1) % (2*N) == 0 for NEGACYCLIC NTT.
// Old code generated (q-1) % N == 0 primes (cyclic), wrong for FHE.
static vector<uint64_t> gen_negacyclic_ntt_primes(uint32_t N, int count) {
    vector<uint64_t> primes;
    // step must be 2*N so that q = c*(2*N)+1 satisfies (q-1) % (2*N) == 0.
    uint64_t step = 2ULL * N;
    uint64_t c = (1ULL << 56) / step;
    while ((int)primes.size() < count) {
        uint64_t q = c * step + 1;
        if (q > 2 && is_prime(q)) primes.push_back(q);
        c++;
    }
    return primes;
}

static uint64_t mulmod_cpu(uint64_t a, uint64_t b, uint64_t q) {
    return (uint64_t)(((unsigned __int128)a * b) % q);
}
static uint64_t powmod_cpu(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t r=1; base%=mod;
    while(exp>0){if(exp&1)r=(uint64_t)(((unsigned __int128)r*base)%mod);base=(uint64_t)(((unsigned __int128)base*base)%mod);exp>>=1;}
    return r;
}
static uint32_t bit_rev(uint32_t x, uint32_t log2n){
    uint32_t r=0;
    for(uint32_t i=0;i<log2n;i++){r=(r<<1)|(x&1);x>>=1;}
    return r;
}

// FIX: CPU negacyclic NTT using a 2N-th primitive root psi.
// Old code used w = g^((q-1)/N) which is an N-th root → cyclic NTT.
// New code uses psi = g^((q-1)/(2N)), psi^N = -1 mod q → negacyclic NTT.
//
// Implementation: twisted NTT.
//   1. Pre-multiply: a[k] *= psi^k  (twist)
//   2. Run standard cyclic NTT with w = psi^2 (which is an N-th root)
//   3. Result is the negacyclic NTT of the original a.
//
// INTT: reverse the steps (cyclic INTT with w_inv, then divide by psi^k).
static uint64_t find_psi(uint64_t q, uint32_t N) {
    uint64_t phi = q - 1;
    for (uint64_t g = 2; g < q; g++) {
        bool ok = true; uint64_t tmp = phi;
        for (uint64_t p = 2; p*p <= tmp; p++) {
            if (tmp%p==0) {
                if (powmod_cpu(g, phi/p, q)==1){ok=false;break;}
                while(tmp%p==0)tmp/=p;
            }
        }
        if (ok && tmp>1 && powmod_cpu(g,phi/tmp,q)==1) ok=false;
        if (ok) return powmod_cpu(g, (q-1)/(2ULL*N), q);
    }
    return 0;
}

static void cpu_negacyclic_ntt(vector<uint64_t>& a, uint64_t q, uint64_t psi) {
    uint32_t N=a.size(), log2N=0;
    while((1u<<log2N)<N) log2N++;
    // Step 1: twist by psi^k.
    uint64_t pk=1;
    for(uint32_t k=0;k<N;k++){
        a[k]=mulmod_cpu(a[k],pk,q);
        pk=mulmod_cpu(pk,psi,q);
    }
    // Step 2: standard cyclic NTT with w = psi^2 (N-th root).
    uint64_t w = mulmod_cpu(psi,psi,q);
    // bit-reversal.
    for(uint32_t i=0;i<N;i++){uint32_t j=bit_rev(i,log2N);if(j>i)swap(a[i],a[j]);}
    // DIT.
    for(uint32_t half_m=1;half_m<=N/2;half_m<<=1){
        uint64_t wn=powmod_cpu(w,N/(2*half_m),q);
        for(uint32_t k=0;k<N;k+=2*half_m){
            uint64_t wj=1;
            for(uint32_t j=0;j<half_m;j++){
                uint64_t u=a[k+j], v=mulmod_cpu(a[k+j+half_m],wj,q);
                a[k+j]=(u+v>=q)?(u+v-q):(u+v);
                a[k+j+half_m]=(u>=v)?(u-v):(u+q-v);
                wj=mulmod_cpu(wj,wn,q);
            }
        }
    }
}

static void cpu_negacyclic_intt(vector<uint64_t>& a, uint64_t q, uint64_t psi) {
    uint32_t N=a.size();
    uint64_t w=mulmod_cpu(psi,psi,q);
    uint64_t w_inv=powmod_cpu(w,q-2,q);
    // Inverse cyclic NTT with w_inv.
    uint32_t log2N=0; while((1u<<log2N)<N) log2N++;
    for(uint32_t i=0;i<N;i++){uint32_t j=bit_rev(i,log2N);if(j>i)swap(a[i],a[j]);}
    for(uint32_t half_m=1;half_m<=N/2;half_m<<=1){
        uint64_t wn=powmod_cpu(w_inv,N/(2*half_m),q);
        for(uint32_t k=0;k<N;k+=2*half_m){
            uint64_t wj=1;
            for(uint32_t j=0;j<half_m;j++){
                uint64_t u=a[k+j], v=mulmod_cpu(a[k+j+half_m],wj,q);
                a[k+j]=(u+v>=q)?(u+v-q):(u+v);
                a[k+j+half_m]=(u>=v)?(u-v):(u+q-v);
                wj=mulmod_cpu(wj,wn,q);
            }
        }
    }
    uint64_t n_inv=powmod_cpu(N,q-2,q);
    for(auto& x:a) x=mulmod_cpu(x,n_inv,q);
    // Undo twist: multiply by psi_inv^k.
    uint64_t psi_inv=powmod_cpu(psi,q-2,q);
    uint64_t pk=1;
    for(uint32_t k=0;k<N;k++){
        a[k]=mulmod_cpu(a[k],pk,q);
        pk=mulmod_cpu(pk,psi_inv,q);
    }
}

int main() {
    const uint32_t N=32768, NUM_TOWERS=16, NUM_THREADS=8;
    cout<<"======================================================\n";
    cout<<"[*] Negacyclic NTT GPU Verification Engine\n";
    cout<<"[*] N="<<N<<" towers="<<NUM_TOWERS<<" threads="<<NUM_THREADS<<"\n";
    cout<<"======================================================\n";

    // FIX: use negacyclic-NTT-friendly primes ((q-1) % 2N == 0).
    auto primes = gen_negacyclic_ntt_primes(N, NUM_TOWERS);
    for(int i=0;i<NUM_TOWERS;i++)
        cout<<"  Prime["<<i<<"] = "<<primes[i]
            <<"  (q-1)%"<<2*N<<"="<<(primes[i]-1)%(2ULL*N)<<"\n";

    bool global_ok = true;

    // ── TEST 1: pointwise RNS multiply (no NTT) ───────────────────────────────
    cout<<"\n[TEST 1] Pointwise RNS multiply ("<<NUM_THREADS<<" OMP threads)\n";
    auto t0 = clk::now();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(dynamic,1)
    for(int tid=0;tid<NUM_THREADS;tid++) {
        mt19937_64 lrng(tid*1000+1);
        vector<vector<uint64_t>> A(NUM_TOWERS,vector<uint64_t>(N));
        vector<vector<uint64_t>> B(NUM_TOWERS,vector<uint64_t>(N));
        vector<vector<uint64_t>> R(NUM_TOWERS,vector<uint64_t>(N,0));
        for(int t=0;t<NUM_TOWERS;t++){uint64_t q=primes[t];for(uint32_t k=0;k<N;k++){A[t][k]=lrng()%q;B[t][k]=lrng()%q;}}
        vector<const uint64_t*> pA(NUM_TOWERS),pB(NUM_TOWERS); vector<uint64_t*> pR(NUM_TOWERS);
        for(int t=0;t<NUM_TOWERS;t++){pA[t]=A[t].data();pB[t]=B[t].data();pR[t]=R[t].data();}
        gpu_rns_mult_batch_wrapper(pA.data(),pB.data(),pR.data(),primes.data(),N,NUM_TOWERS);
        bool ok=true;
        for(int t=0;t<NUM_TOWERS&&ok;t++){
            uint64_t q=primes[t];
            for(uint32_t k=0;k<N&&ok;k++){
                uint64_t want=mulmod_cpu(A[t][k],B[t][k],q);
                if(R[t][k]!=want){
                    cout<<"  MISMATCH thread="<<tid<<" tower="<<t<<" idx="<<k
                        <<" got="<<R[t][k]<<" want="<<want<<"\n";
                    ok=false;
                }
            }
        }
        if(ok) cout<<"[+] Thread "<<tid<<" pointwise OK\n";
        else {
            cout<<"[-] Thread "<<tid<<" FAILED\n";
            #pragma omp atomic write
            global_ok = false;
        }
    }
    cout<<"  elapsed="<<chrono::duration<double,milli>(clk::now()-t0).count()<<"ms\n";

    // ── TEST 2: negacyclic NTT polynomial multiply ────────────────────────────
    // FIX: CPU reference now computes NEGACYCLIC convolution:
    //   c[k] = sum_{i+j=k mod N, wrapping with sign flip} a[i]*b[j]
    // Specifically: if i+j >= N, the coefficient gets a factor of -1 (mod q).
    // Old code computed cyclic (no sign flip) -- wrong for FHE.
    cout<<"\n[TEST 2] Negacyclic NTT polynomial multiply (N=16, q from 2N-friendly primes)\n";
    {
        // Use a small N=16 prime satisfying (q-1) % 32 == 0 for clarity.
        auto small_primes = gen_negacyclic_ntt_primes(16, 1);
        uint64_t q = small_primes[0];
        uint32_t Ns = 16;
        uint64_t psi = find_psi(q, Ns);
        cout<<"  q="<<q<<" psi="<<psi<<"\n";
        cout<<"  psi^"<<Ns<<" mod q = "<<powmod_cpu(psi,Ns,q)
            <<" (expect "<<q-1<<" i.e. -1 mod q)\n";
        cout<<"  psi^"<<2*Ns<<" mod q = "<<powmod_cpu(psi,2*Ns,q)<<" (expect 1)\n";

        mt19937_64 rng2(99);
        vector<uint64_t> a(Ns), b(Ns);
        for(auto& x:a) x=rng2()%q;
        for(auto& x:b) x=rng2()%q;

        // CPU negacyclic convolution reference:
        // c[k] = sum_{i} a[i] * b[(k-i) mod N] * (-1 if (k-i) wraps
        vector<uint64_t> ref(Ns, 0);
        for(uint32_t i=0;i<Ns;i++) {
            for(uint32_t j=0;j<Ns;j++) {
                uint32_t idx = (i+j) % Ns;
                uint64_t term = mulmod_cpu(a[i], b[j], q);
                if (i+j >= Ns) {
                    // Negacyclic wrap: multiply by -1 mod q.
                    term = (term == 0) ? 0 : q - term;
                }
                ref[idx] = (ref[idx] + term >= q) ? ref[idx] + term - q : ref[idx] + term;
            }
        }

        // CPU NTT negacyclic convolution (should match ref).
        vector<uint64_t> ca=a, cb=b;
        cpu_negacyclic_ntt(ca, q, psi);
        cpu_negacyclic_ntt(cb, q, psi);
        vector<uint64_t> cc(Ns);
        for(uint32_t i=0;i<Ns;i++) cc[i]=mulmod_cpu(ca[i],cb[i],q);
        cpu_negacyclic_intt(cc, q, psi);

        cout<<"  CPU negacyclic NTT conv vs ref:\n";
        bool cpu_ok=true;
        for(uint32_t i=0;i<Ns;i++) {
            if(cc[i]!=ref[i]){cout<<"    MISMATCH idx="<<i<<" got="<<cc[i]<<" want="<<ref[i]<<"\n";cpu_ok=false;}
        }
        if(cpu_ok) cout<<"  [+] CPU negacyclic NTT self-consistent\n";
        else       cout<<"  [-] CPU negacyclic NTT BUG\n";

        // GPU negacyclic convolution.
        vector<uint64_t> r(Ns, 0);
        const uint64_t* pA2=a.data(); const uint64_t* pB2=b.data(); uint64_t* pR2=r.data();
        gpu_poly_mult_wrapper(&pA2, &pB2, &pR2, &q, Ns, 1);

        cout<<"  GPU vs negacyclic reference:\n";
        bool gpu_ok=true;
        for(uint32_t i=0;i<Ns;i++) {
            cout<<"    ["<<i<<"] gpu="<<r[i]<<" cpu_ntt="<<cc[i]<<" ref="<<ref[i];
            if(r[i]!=ref[i]){cout<<" MISMATCH";gpu_ok=false;}
            cout<<"\n";
        }
        if(gpu_ok) cout<<"[+] GPU negacyclic NTT convolution OK\n";
        else       {cout<<"[-] GPU negacyclic NTT FAILED\n"; global_ok=false;}
    }


    // ── TEST 3: pointwise RNS throughput (N=32768) ───────────────────────────
    cout<<"\n[TEST 3] Pointwise RNS throughput (N=32768, 16 towers)\n";
    {
        mt19937_64 rng(42);
        vector<vector<uint64_t>> A(NUM_TOWERS,vector<uint64_t>(N));
        vector<vector<uint64_t>> B(NUM_TOWERS,vector<uint64_t>(N));
        vector<vector<uint64_t>> R(NUM_TOWERS,vector<uint64_t>(N,0));
        for(int t=0;t<NUM_TOWERS;t++){uint64_t q=primes[t];for(uint32_t k=0;k<N;k++){A[t][k]=rng()%q;B[t][k]=rng()%q;}}
        vector<const uint64_t*> pA(NUM_TOWERS),pB(NUM_TOWERS); vector<uint64_t*> pR(NUM_TOWERS);
        for(int t=0;t<NUM_TOWERS;t++){pA[t]=A[t].data();pB[t]=B[t].data();pR[t]=R[t].data();}
        gpu_rns_mult_batch_wrapper(pA.data(),pB.data(),pR.data(),primes.data(),N,NUM_TOWERS);
        const int ITERS=20;
        auto ts=clk::now();
        for(int i=0;i<ITERS;i++)
            gpu_rns_mult_batch_wrapper(pA.data(),pB.data(),pR.data(),primes.data(),N,NUM_TOWERS);
        double ms=chrono::duration<double,milli>(clk::now()-ts).count()/ITERS;
        printf("  RNS N=32768:  %.2f ms/op  (%.1f M coeff-mults/s)\n",
               ms, (double)(NUM_TOWERS*N)/(ms*1e3));
    }

    // ── TEST 4: large-ring RNS throughput (N=65536) ──────────────────────────
    cout<<"\n[TEST 4] Large-ring pointwise RNS throughput (N=65536, 16 towers)\n";
    {
        uint32_t Nbig = 65536;
        auto big_primes = gen_negacyclic_ntt_primes(Nbig, NUM_TOWERS);
        mt19937_64 rng(99);
        vector<vector<uint64_t>> A(NUM_TOWERS,vector<uint64_t>(Nbig));
        vector<vector<uint64_t>> B(NUM_TOWERS,vector<uint64_t>(Nbig));
        vector<vector<uint64_t>> R(NUM_TOWERS,vector<uint64_t>(Nbig,0));
        for(int t=0;t<NUM_TOWERS;t++){uint64_t q=big_primes[t];for(uint32_t k=0;k<Nbig;k++){A[t][k]=rng()%q;B[t][k]=rng()%q;}}
        vector<const uint64_t*> pA(NUM_TOWERS),pB(NUM_TOWERS); vector<uint64_t*> pR(NUM_TOWERS);
        for(int t=0;t<NUM_TOWERS;t++){pA[t]=A[t].data();pB[t]=B[t].data();pR[t]=R[t].data();}
        // warmup
        gpu_rns_mult_batch_wrapper(pA.data(),pB.data(),pR.data(),big_primes.data(),Nbig,NUM_TOWERS);
        const int ITERS=10;
        auto ts=clk::now();
        for(int i=0;i<ITERS;i++)
            gpu_rns_mult_batch_wrapper(pA.data(),pB.data(),pR.data(),big_primes.data(),Nbig,NUM_TOWERS);
        double ms=chrono::duration<double,milli>(clk::now()-ts).count()/ITERS;
        printf("  RNS N=65536:  %.2f ms/op  (%.1f M coeff-mults/s)\n",
               ms, (double)(NUM_TOWERS*Nbig)/(ms*1e3));
    }

    // ── TEST 5: NTT poly multiply throughput (N=32768) ────────────────────────
    cout<<"\n[TEST 5] NTT polynomial multiply throughput (N=32768, 16 towers)\n";
    {
        mt19937_64 rng(77);
        vector<vector<uint64_t>> A(NUM_TOWERS,vector<uint64_t>(N));
        vector<vector<uint64_t>> B(NUM_TOWERS,vector<uint64_t>(N));
        vector<vector<uint64_t>> R(NUM_TOWERS,vector<uint64_t>(N,0));
        for(int t=0;t<NUM_TOWERS;t++){uint64_t q=primes[t];for(uint32_t k=0;k<N;k++){A[t][k]=rng()%q;B[t][k]=rng()%q;}}
        vector<const uint64_t*> pA(NUM_TOWERS),pB(NUM_TOWERS); vector<uint64_t*> pR(NUM_TOWERS);
        for(int t=0;t<NUM_TOWERS;t++){pA[t]=A[t].data();pB[t]=B[t].data();pR[t]=R[t].data();}
        // warmup
        gpu_poly_mult_wrapper(pA.data(),pB.data(),pR.data(),primes.data(),N,NUM_TOWERS);
        const int ITERS=10;
        auto ts=clk::now();
        for(int i=0;i<ITERS;i++)
            gpu_poly_mult_wrapper(pA.data(),pB.data(),pR.data(),primes.data(),N,NUM_TOWERS);
        double ms=chrono::duration<double,milli>(clk::now()-ts).count()/ITERS;
        printf("  NTT poly mult N=32768:  %.2f ms/op  (%.1f M coeff-mults/s)\n",
               ms, (double)(NUM_TOWERS*N)/(ms*1e3));
    }

    // ── TEST 6: NTT poly multiply throughput (N=65536) ───────────────────────
    cout<<"\n[TEST 6] NTT polynomial multiply throughput (N=65536, 16 towers)\n";
    {
        uint32_t Nbig = 65536;
        auto big_primes = gen_negacyclic_ntt_primes(Nbig, NUM_TOWERS);
        mt19937_64 rng(55);
        vector<vector<uint64_t>> A(NUM_TOWERS,vector<uint64_t>(Nbig));
        vector<vector<uint64_t>> B(NUM_TOWERS,vector<uint64_t>(Nbig));
        vector<vector<uint64_t>> R(NUM_TOWERS,vector<uint64_t>(Nbig,0));
        for(int t=0;t<NUM_TOWERS;t++){uint64_t q=big_primes[t];for(uint32_t k=0;k<Nbig;k++){A[t][k]=rng()%q;B[t][k]=rng()%q;}}
        vector<const uint64_t*> pA(NUM_TOWERS),pB(NUM_TOWERS); vector<uint64_t*> pR(NUM_TOWERS);
        for(int t=0;t<NUM_TOWERS;t++){pA[t]=A[t].data();pB[t]=B[t].data();pR[t]=R[t].data();}
        gpu_poly_mult_wrapper(pA.data(),pB.data(),pR.data(),big_primes.data(),Nbig,NUM_TOWERS);
        const int ITERS=10;
        auto ts=clk::now();
        for(int i=0;i<ITERS;i++)
            gpu_poly_mult_wrapper(pA.data(),pB.data(),pR.data(),big_primes.data(),Nbig,NUM_TOWERS);
        double ms=chrono::duration<double,milli>(clk::now()-ts).count()/ITERS;
        printf("  NTT poly mult N=65536:  %.2f ms/op  (%.1f M coeff-mults/s)\n",
               ms, (double)(NUM_TOWERS*Nbig)/(ms*1e3));
    }

    cout<<"\n======================================================\n";
    cout<<(global_ok?"[PASS] All tests passed":"[FATAL] Tests FAILED")<<"\n";
    cout<<"======================================================\n";
    return global_ok ? 0 : 1;
}

