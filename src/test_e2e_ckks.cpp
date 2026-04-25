#include <openfhe.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace lbcrypto;
using clk = std::chrono::high_resolution_clock;
static double approx_error(const std::vector<std::complex<double>>& a,
                            const std::vector<std::complex<double>>& b) {
    double e = 0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); i++)
        e = std::max(e, std::abs(a[i] - b[i]));
    return e;
}
int main() {
    std::cout << "======================================================\n";
    std::cout << "[*] OpenFHE NVIDIA GPU HAL -- End-to-End CKKS Test\n";
    std::cout << "======================================================\n\n";
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(5);
    params.SetScalingModSize(50);
    params.SetBatchSize(16384);
    params.SetRingDim(32768);
    auto t0 = clk::now();
    CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
    cc->Enable(PKE); cc->Enable(LEVELEDSHE);
    auto kp = cc->KeyGen();
    cc->EvalMultKeyGen(kp.secretKey);
    std::cout << "[1] Key generation: " << std::fixed << std::setprecision(2)
              << std::chrono::duration<double,std::milli>(clk::now()-t0).count() << " ms\n";
    size_t slots = 16384;
    std::vector<double> x(slots), y(slots);
    for (size_t i=0;i<slots;i++){x[i]=0.1*(i%10+1);y[i]=0.5+0.01*(i%7);}
    std::vector<std::complex<double>> expected(slots);
    for (size_t i=0;i<slots;i++) expected[i]=x[i]*y[i];
    auto t1=clk::now();
    auto ctx=cc->Encrypt(kp.publicKey,cc->MakeCKKSPackedPlaintext(x));
    auto cty=cc->Encrypt(kp.publicKey,cc->MakeCKKSPackedPlaintext(y));
    std::cout << "[2] Encrypt: " << std::chrono::duration<double,std::milli>(clk::now()-t1).count() << " ms\n";
    auto t2=clk::now();
    auto ct_result=cc->EvalMult(ctx,cty);
    std::cout << "[3] EvalMult: " << std::chrono::duration<double,std::milli>(clk::now()-t2).count() << " ms\n";
    auto t3=clk::now();
    Plaintext pt_out;
    cc->Decrypt(kp.secretKey,ct_result,&pt_out);
    pt_out->SetLength(slots);
    std::cout << "[4] Decrypt: " << std::chrono::duration<double,std::milli>(clk::now()-t3).count() << " ms\n";
    auto got=pt_out->GetCKKSPackedValue();
    double err=approx_error(got,expected);
    for (int i=0;i<8;i++)
        printf("    [%d] exp=%.8f got=%.8f err=%.2e\n",i,expected[i].real(),got[i].real(),std::abs(got[i]-expected[i]));
    std::cout << "\nMax error: " << err << "\n";
    bool pass=(err<1e-6);
    std::cout << (pass?"[PASS]\n":"[FAIL]\n");
    return pass?0:1;
}
