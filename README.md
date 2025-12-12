# addfftHQC
HQC with addFFT polynomial multiplication

## Contents

#### HQC implementations: 
1.  `hqc-official/` : contains the official x86_64 and reference code cloned from the official website (https://gitlab.com/pqc-hqc/hqc/) (rev: d622142a50f3ce6b6e1f5b15a5119d96c67194e0 )
```
hqc-official/
├── CMakeLists.txt           # Project-specific build configuration
├── lib/                     # Libraries
|    └── fips202/            # Reference implementation for FIPS202
└── src/                     # Implementation files
     ├── common/             # Common files shared across all implementations
     ├── ref/                # Reference implementation
     └── x86_64/             # Optimized implementation for x86_64 (AVX2 with PCLMULDQD)
```

2.  `hqc-fft/` : contains our optimization for x86_64, GFNI, Apple M1 and Arm Cortex-a72. We use symbolic link for unchanged files in the official package.
```
hqc-fft/
├── Makefile                 # Project-specific build configuration
├── benchmark/               # Benchmark tools. (a symblock link to benchmarktool/)
├── lib/                     # Libraries
|    ├── fips202/            # Reference implementation for FIPS202. (a symblock link to hqc-official/lib/fips202/)
|    ├── bitpolymul_a72/     # Optimized polynomial multiplication for Arm Cortex-a72 platform. (a symblock link to bitpolymul/gf256x2_neon/)
|    ├── bitpolymul_gfni/    # Optimized polynomial multiplication for x86 GFNI platform. (a symblock link to bitpolymul/gf256t4_gfni/)
|    ├── bitpolymul_m1/      # Optimized polynomial multiplication for Apple M1 platform. (a symblock link to bitpolymul/gf264_neon/)
|    ├── bitpolymul_ref/     # Refernece implementation for polynomial multiplication on uint32_t platform. (a symblock link to bitpolymul/gf256x2_ref/)
|    └── bitpolymul_x86/     # Optimized polynomial multiplication for AVX2 platform. (a symblock link to bitpolymul/gf264_avx2/)
├── src/                     # Implementation files
|    ├── a72/                # Optimized HQC implementation for Arm Cortex-a72. (Armv8 without crypto extension)
|    ├── armv8/              # Optimized HQC implementation for Armv8. (with crypto extension)
|    ├── cachekey/           # Interface, structures, and implementations for supporting cached key functionalities.
|    ├── code_x86/           # Optimized RS encoder and RM decoder on x86 platforms. (AVX2 and GFNI)
|    ├── common/             # Common files shared across all implementations. (a symblock link to hqc-official/src/common/)
|    ├── ref/                # Reference implementation. (a symblock link to hqc-official/src/ref/)
|    └── x86_64/             # Optimized implementation for x86_64. (a symblock link to hqc-official/src/x86_64/)
└──  tests/                  # Unit-testers and benchmarkers.
```


#### Libraries for multiplication for bit-polynomials : bitpolymul/

```
bitpolymul/                # Libraries for multiplication for bit-polynomials
├── gf256t4_avx2gfni/      # AVX2+GFNI optimization for Frobenius addFFT on GF(256^4).
├── gf256t4_ref/           # Reference implementation for Frobenius addFFT on GF(256^4).
├── gf256x2_neon/          # NEON optimization for Kronecker substitution on GF(256^2).
├── gf256x2_ref/           # Refernece implementation for Kronecker substitution on GF(256^2).
├── gf264_avx2/            # AVX2+PCLMULQDQ optimization for Frobenius addFFT on GF(2^64).
├── gf264_neon/            # NEON optimization for Frobenius addFFT on GF(2^64).
└── gf264_ref/             # Refernece implementation for Frobenius addFFT on GF(2^64).
```


#### Benchmark tools : benchmarktool/

## Makefile guide

This guide explains how to use the Makefile to build and test different configurations of the implementation.

### Basic Usage

```bash
cd hqc-fft
make TARGET=<method> PROJ=<parameter>
```

- The default setting is `TARGET=fft_ref` and `PROJ=hqc-1` which runs the reference (portable) FFT implementation.

### Configuration Options

#### PROJ - Parameter Sets

Choose one of three HQC parameter sets:

- `hqc-1` - HQC parameter set 1
- `hqc-3` - HQC parameter set 3
- `hqc-5` - HQC parameter set 5

#### TARGET - The algorithm and the target platform 

Choose between two polynomial multiplication and encode/decode implementations:

- `ref` - The official reference code, run on every platform
- `x86_hqcteam` - Optimized x86_64 code in HQC official package
- `m1_hqcport` - Our porting of the optimized HQC official package on Mac M1
- `a72_hqcport` - Our porting of the optimized HQC official package on Raspberry Pi4(Cortex-a72)
- `fft_ref` - Our proposed method reference code, run on every platform
- `fft_x86` - Our proposed method on x86_64
- `fft_gfni` - Our proposed method on x86_64 with GFNI instructions
- `fft_m1` - Our proposed method on Mac m1
- `fft_a72` - Our proposed method on Raspberry Pi4(Cortex-a72)

#### Testing

After compiling, run the following command to test and benchmark defferent components.

```bash
sudo ./bin/gf2x-test
sudo ./bin/code-test
sudo ./bin/hqc-test
```


## Usage
We also provide benchmark files `benchmark.sh` for simple usage.
This will automatically record benchmark results in text files.

Under main folder addfftHQC/
```bash
./newbench.sh <target> <textfile.txt>
```
The `target` parameter follows the target algorithm and platform in previous sections. 

For example:
```
./newbench.sh fft_x86 result_fftx86.txt
```
Alternatively, if your computer has the GFNI instruction set:
```
./newbench.sh fft_gfni result_fftgfni.txt
```
If you are on Mac M1:
```
./newbench.sh m1_fft result_m1fft.txt
```
Or, the default is (benchmark on x86):
```
./newbench.sh ref results.txt
```

