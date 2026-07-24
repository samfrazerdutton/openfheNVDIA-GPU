======================================================
[*] CKKS EvalMult -- paired-difference benchmark
    modes interleaved per rep in one process; reported
    statistic is median(delta) with a 95% bootstrap CI.
    reps=50 warmup=10/mode
======================================================

depth=5 ring=32768 towers=7
    absolute        cpu=64.00ms  gpu=67.44ms  gpu-ks=73.78ms  
    IQR             cpu=24.13ms  gpu=27.17ms  gpu-ks=21.24ms  
    gpu - cpu            2.58 ms  [  -0.33,   5.67]      4.0%   not significant
    gpu-ks - cpu        10.70 ms  [   6.18,  15.43]     16.7%   SLOWER
    gpu-ks - gpu         9.53 ms  [   5.76,  14.45]     14.1%   SLOWER

depth=10 ring=32768 towers=12
    absolute        cpu=114.25ms  gpu=115.89ms  gpu-ks=120.08ms  
    IQR             cpu=37.69ms  gpu=38.49ms  gpu-ks=25.81ms  
    gpu - cpu            3.25 ms  [  -3.79,   7.58]      2.8%   not significant
    gpu-ks - cpu        10.05 ms  [   2.14,  16.95]      8.8%   SLOWER
    gpu-ks - gpu         7.03 ms  [  -0.04,  22.13]      6.1%   not significant

depth=5 ring=65536 towers=7
    absolute        cpu=92.38ms  gpu=99.24ms  gpu-ks=105.77ms  
    IQR             cpu=25.82ms  gpu=30.03ms  gpu-ks=25.75ms  
    gpu - cpu            3.68 ms  [  -1.44,   8.42]      4.0%   not significant
    gpu-ks - cpu        15.82 ms  [  10.31,  20.31]     17.1%   SLOWER
    gpu-ks - gpu        12.88 ms  [   4.93,  16.08]     13.0%   SLOWER

depth=10 ring=65536 towers=12
    absolute        cpu=194.12ms  gpu=207.22ms  gpu-ks=211.05ms  
    IQR             cpu=43.13ms  gpu=46.62ms  gpu-ks=37.08ms  
    gpu - cpu            3.56 ms  [  -4.63,  17.65]      1.8%   not significant
    gpu-ks - cpu        11.02 ms  [  -1.35,  22.53]      5.7%   not significant
    gpu-ks - gpu         6.65 ms  [  -2.86,  10.69]      3.2%   not significant

depth=15 ring=65536 towers=17
    absolute        cpu=277.65ms  gpu=269.08ms  gpu-ks=294.35ms  
    IQR             cpu=68.67ms  gpu=59.34ms  gpu-ks=39.40ms  
    gpu - cpu            5.70 ms  [  -8.90,  21.65]      2.1%   not significant
    gpu-ks - cpu        23.94 ms  [  10.53,  42.35]      8.6%   SLOWER
    gpu-ks - gpu        15.27 ms  [   5.61,  31.74]      5.7%   SLOWER

A delta is meaningful only when its CI excludes zero. The
absolute medians are shown for scale; do not compare them
across separate runs of this binary.
