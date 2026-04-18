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

static vector<uint64_t> gen_ntt_primes(uint32_t N, int count) {
    vector<uint64_t> primes;
    uint64_t step = 2ULL*N;
    uint64_t c = (1ULL<<56)/step;
    while((int)primes.size()<count){uint64_t q=c*step+1;if(q>2&&is_prime(q))primes.push_back(q);c++;}
    return primes;
}

static uint64_t mulmod_cpu(uint64_t a,uint64_t b,uint64_t q){
    return (uint64_t)(((unsigned __int128)a*b)%q);
}
static uint64_t powmod_cpu(uint64_t base,uint64_t exp,uint64_t mod){
    uint64_t r=1;base%=mod;
    while(exp>0){if(exp&1)r=(uint64_t)(((unsigned __int128)r*base)%mod);base=(uint64_t)(((unsigned __int128)base*base)%mod);exp>>=1;}
    return r;
}

// ── CPU reference NTT (Cooley-Tukey DIT, bit-reversed input) ─────────────────
static uint32_t bit_rev(uint32_t x, uint32_t log2n){
    uint32_t r=0;
    for(uint32_t i=0;i<log2n;i++){r=(r<<1)|(x&1);x>>=1;}
    return r;
}
static void cpu_ntt(vector<uint64_t>& a, uint64_t q, uint64_t w_prim){
    uint32_t N=a.size(), log2N=0;
    while((1u<<log2N)<N)log2N++;
    // bit-reversal
    for(uint32_t i=0;i<N;i++){uint32_t j=bit_rev(i,log2N);if(j>i)swap(a[i],a[j]);}
    // DIT
    for(uint32_t half_m=1;half_m<=N/2;half_m<<=1){
        // w for this stage: w_prim^(N/(2*half_m))
        uint64_t wn=powmod_cpu(w_prim,N/(2*half_m),q);
        for(uint32_t k=0;k<N;k+=2*half_m){
            uint64_t wj=1;
            for(uint32_t j=0;j<half_m;j++){
                uint64_t u=a[k+j];
                uint64_t v=mulmod_cpu(a[k+j+half_m],wj,q);
                a[k+j]       =(u+v>=q)?(u+v-q):(u+v);
                a[k+j+half_m]=(u>=v)?(u-v):(u+q-v);
                wj=mulmod_cpu(wj,wn,q);
            }
        }
    }
}
static void cpu_intt(vector<uint64_t>& a, uint64_t q, uint64_t w_prim){
    uint32_t N=a.size();
    uint64_t w_inv=powmod_cpu(w_prim,q-2,q);
    cpu_ntt(a,q,w_inv);
    uint64_t n_inv=powmod_cpu(N,q-2,q);
    for(auto& x:a) x=mulmod_cpu(x,n_inv,q);
}
// Find primitive 2N-th root of unity mod q
static uint64_t find_w(uint64_t q, uint32_t N){
    uint64_t phi=q-1;
    for(uint64_t g=2;g<q;g++){
        uint64_t tmp=phi; bool ok=true;
        for(uint64_t p=2;p*p<=tmp;p++){
            if(tmp%p==0){if(powmod_cpu(g,phi/p,q)==1){ok=false;break;}while(tmp%p==0)tmp/=p;}
        }
        if(ok&&tmp>1&&powmod_cpu(g,phi/tmp,q)==1)ok=false;
        if(ok){return powmod_cpu(g,(q-1)/N,q);}
    }
    return 0;
}

int main(){
    const uint32_t N=32768, NUM_TOWERS=16, NUM_THREADS=8;
    cout<<"======================================================\n";
    cout<<"[*] Duality-Grade OpenFHE GPU Verification Engine\n";
    cout<<"[*] N="<<N<<" towers="<<NUM_TOWERS<<" threads="<<NUM_THREADS<<"\n";
    cout<<"======================================================\n";

    auto primes=gen_ntt_primes(N,NUM_TOWERS);
    for(int i=0;i<NUM_TOWERS;i++) cout<<"  Prime["<<i<<"] = "<<primes[i]<<"\n";

    bool global_ok=true;

    // ── TEST 1 ────────────────────────────────────────────────────────────────
    cout<<"\n[TEST 1] Pointwise RNS multiply ("<<NUM_THREADS<<" OMP threads)\n";
    auto t0=clk::now();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(dynamic,1)
    for(int tid=0;tid<NUM_THREADS;tid++){
        mt19937_64 lrng(tid*1000+1);
        vector<vector<uint64_t>> A(NUM_TOWERS,vector<uint64_t>(N));
        vector<vector<uint64_t>> B(NUM_TOWERS,vector<uint64_t>(N));
        vector<vector<uint64_t>> R(NUM_TOWERS,vector<uint64_t>(N,0));
        for(int t=0;t<NUM_TOWERS;t++){uint64_t q=primes[t];for(uint32_t k=0;k<N;k++){A[t][k]=lrng()%q;B[t][k]=lrng()%q;}}
        vector<const uint64_t*> pA(NUM_TOWERS),pB(NUM_TOWERS); vector<uint64_t*> pR(NUM_TOWERS);
        for(int t=0;t<NUM_TOWERS;t++){pA[t]=A[t].data();pB[t]=B[t].data();pR[t]=R[t].data();}
        gpu_rns_mult_batch_wrapper(pA.data(),pB.data(),pR.data(),primes.data(),N,NUM_TOWERS);
        bool ok=true;
        for(int t=0;t<NUM_TOWERS&&ok;t++){uint64_t q=primes[t];
            for(uint32_t k=0;k<N&&ok;k++){uint64_t want=mulmod_cpu(A[t][k],B[t][k],q);
                if(R[t][k]!=want){cout<<"  MISMATCH thread="<<tid<<" tower="<<t<<" idx="<<k<<" got="<<R[t][k]<<" want="<<want<<"\n";ok=false;}}}
        if(ok)cout<<"[+] Thread "<<tid<<" pointwise OK\n";
        else{
            cout<<"[-] Thread "<<tid<<" FAILED\n";
            #pragma omp atomic write
            global_ok=false;
        }
    }
    cout<<"  elapsed="<<chrono::duration<double,milli>(clk::now()-t0).count()<<"ms\n";

    // ── TEST 2: CPU trace to find exact divergence point ─────────────────────
    cout<<"\n[TEST 2] NTT trace (N=16, q=7681)\n";
    {
        uint64_t q=7681; uint32_t Ns=16;
        uint64_t w=find_w(q,Ns);
        cout<<"  w (primitive 2N root) = "<<w<<"\n";
        cout<<"  w^"<<Ns<<"  mod q = "<<powmod_cpu(w,Ns,q)<<" (expect "<<q-1<<")\n";
        cout<<"  w^"<<2*Ns<<" mod q = "<<powmod_cpu(w,2*Ns,q)<<" (expect 1)\n";

        mt19937_64 rng2(99);
        vector<uint64_t> a(Ns),b(Ns);
        for(auto& x:a)x=rng2()%q;
        for(auto& x:b)x=rng2()%q;

        // CPU cyclic convolution reference
        vector<uint64_t> ref(Ns,0);
        for(uint32_t i=0;i<Ns;i++)
            for(uint32_t j=0;j<Ns;j++)
                ref[(i+j)%Ns]=(uint64_t)((ref[(i+j)%Ns]+(unsigned __int128)a[i]*b[j])%q);

        // CPU NTT convolution (should match ref)
        vector<uint64_t> ca=a, cb=b;
        cpu_ntt(ca,q,w); cpu_ntt(cb,q,w);
        vector<uint64_t> cc(Ns);
        for(uint32_t i=0;i<Ns;i++) cc[i]=mulmod_cpu(ca[i],cb[i],q);
        cpu_intt(cc,q,w);

        cout<<"  CPU NTT conv vs ref:\n";
        bool cpu_ok=true;
        for(uint32_t i=0;i<Ns;i++){
            if(cc[i]!=ref[i]){cout<<"    MISMATCH idx="<<i<<" got="<<cc[i]<<" want="<<ref[i]<<"\n";cpu_ok=false;}
        }
        if(cpu_ok) cout<<"  [+] CPU NTT self-consistent\n";
        else       cout<<"  [-] CPU NTT BUG (twiddle_gen or cpu_ntt wrong)\n";

        // GPU convolution
        vector<uint64_t> r(Ns,0);
        const uint64_t* pA2=a.data(); const uint64_t* pB2=b.data(); uint64_t* pR2=r.data();
        gpu_poly_mult_wrapper(&pA2,&pB2,&pR2,&q,Ns,1);

        cout<<"  GPU vs CPU NTT result:\n";
        bool gpu_ok=true;
        for(uint32_t i=0;i<Ns;i++){
            cout<<"    ["<<i<<"] gpu="<<r[i]<<" cpu_conv="<<cc[i]<<" ref="<<ref[i];
            if(r[i]!=ref[i]){cout<<" MISMATCH";gpu_ok=false;}
            cout<<"\n";
        }
        if(gpu_ok) cout<<"[+] GPU NTT convolution OK\n";
        else{cout<<"[-] GPU NTT FAILED\n"; global_ok=false;}
    }

    cout<<"\n======================================================\n";
    cout<<(global_ok?"[PASS] All tests passed":"[FATAL] Tests FAILED")<<"\n";
    cout<<"======================================================\n";
    return global_ok?0:1;
}
