# addfftHQC
HQC with addFFT polynomial multiplication

## Contents

#### HQC implementations: 

This directory contains HQC optimization for x86-AVX2, x86-GFNI, Apple M1 and Arm Cortex-a72. We use symbolic link for unchanged files in the official package.
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


#### Benchmark tools : benchmarktool/

## Makefile guide

This guide explains how to use the Makefile to build and test different configurations of the implementation.

### Basic Usage

```bash
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


