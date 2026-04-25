#include <openfhe.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cmath>
using namespace lbcrypto;
using clk = std::chrono::high_resolution_clock;
static double max_err(const std::vector<std::complex<double>>& a,
                      const std::vector<std::complex<double>>& b) {
    double e=0;
    for (size_t i=0;i<std::min(a.size(),b.size());i++) e=std::max(e,std::abs(a[i]-b[i]));
    return e;
}
int main() {
    std::cout << "======================================================\n[*] Phase 3+4 E2E CKKS Benchmark\n======================================================\n";
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(4);
    params.SetScalingModSize(50);
    params.SetBatchSize(4096);
    auto cc=GenCryptoContext(params);
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);
    auto keys=cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    std::vector<std::complex<double>> v1(4096,{1.5,0.0}),v2(4096,{2.5,0.0});
    auto ct1=cc->Encrypt(keys.publicKey,cc->MakeCKKSPackedPlaintext(v1));
    auto ct2=cc->Encrypt(keys.publicKey,cc->MakeCKKSPackedPlaintext(v2));
    auto res=cc->EvalMult(ct1,ct2); // warmup
    auto t0=clk::now();
    for (int i=0;i<10;i++) res=cc->EvalMult(ct1,ct2);
    double ms=std::chrono::duration<double,std::milli>(clk::now()-t0).count()/10.0;
    Plaintext pt_out;
    cc->Decrypt(keys.secretKey,res,&pt_out);
    pt_out->SetLength(4096);
    auto got=pt_out->GetCKKSPackedValue();
    std::vector<std::complex<double>> expected(4096,{3.75,0.0});
    double err=max_err(got,expected);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "EvalMult mean (10 reps): " << ms << " ms\n";
    std::cout << "Max error              : " << err << "\n";
    bool pass=(err<1e-4);
    std::cout << (pass?"[PASS]\n":"[FAIL]\n");
    return pass?0:1;
}
