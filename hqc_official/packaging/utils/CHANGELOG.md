## Changelog (before 2025-08-22)

This file is the changelog for HQC implementations released before 2025-08-22 that are accessible for archive purposes at https://pqc-hqc.org/doc/archive_submissions.zip.
Modifications done after 2025-08-22 are available at https://gitlab.com/pqc-hqc/hqc.

### 2025‑08‑22
- Updated both the reference implementation and the optimized implementation to align with the latest version of the official specifications.
- Revised use of the truncate function to conform to the updated specification.
- Removed dependency on NTL in the reference implementation.
- Made the reference implementation constant-time.
- Introduced a Karatsuba-based algorithm for vector multiplication in the reference implementation.
- Added a default call to prng_init(), ensuring the PRNG is automatically initialized.
- Zeroized sensitive buffers to prevent secrets remaining in memory.
- Fixed unintended compiler-introduced branching in constant-time code within the `compute_elp` function of the Reed-Solomon implementation.
  - Reported by Zhenzhi Lai and Zhiyuan Zhang.

### 2025‑02‑19
- Fixed an implementation bug in both the reference and optimized versions of HQC. An indexing error caused incorrect interpretation of the public key during decapsulation, leading to incorrect shared secrets for malformed ciphertexts.
  - Vulnerability reported by Célian Glénaz and Dahmun Goudarzi; disclosed by Spencer Wilson and Douglas Stebila.

### 2024‑10‑30
- Modified the order of variable sampling in key generation and encryption to improve performance in hardware implementations (based on recommendations from external analysis).
- Changed the key binding hash to include only the first 32 bytes of the public key (instead of the full key), along with the message and salt.
- Updated the implementation and KATs accordingly.
- Performance improvements in AVX2 optimized implementation:
    - Encapsulation improved by ~10–13%.
    - Decapsulation improved by ~3–12%.
    - Example timings (in kilocycles):
        - hqc-128: KeyGen 75, Encaps 177, Decaps 323
        - hqc-192: KeyGen 175, Encaps 404, Decaps 669
        - hqc-256: KeyGen 356, Encaps 799, Decaps 1427

### 2024‑02‑23
- Updated implementation to use Barrett reduction instead of the modulo operator to counter timing attacks.
- No change to design or parameters; theoretical IND-CPA proof updated.

### 2023‑04‑30
- Adopted the HHK transform with implicit rejection for CCA2 security.
- Implementation and KATs updated accordingly.

### 2022‑10‑01
- Introduced a public salt in ciphertexts across all security levels to prevent multi-ciphertext attacks in the deterministic variant of HQC-128.
    - θ = SHAKE256-512(m∥pk∥salt)
- Incorporated a countermeasure (Algorithm 5 from Sendrier) to mitigate timing attacks related to small weight word generation.
- Added a constant-time pure C (non-optimized) implementation.
- Hardware implementation still lacks some mitigations; updated version pending.
- Performance summary (in kilocycles):
    - hqc-128: KeyGen 87, Encaps 204, Decaps 362
    - hqc-192: KeyGen 204, Encaps 465, Decaps 755
    - hqc-256: KeyGen 409, Encaps 904, Decaps 1505

### 2021‑06‑06
- Replaced `randombytes` and `seedexpander` with KECCAK-based domain separation and randomness generation.
- Released full HLS-compatible hardware implementations in two variants: performance-oriented and compact.
- Introduced a unified hardware-software design producing identical outputs.

### 2020‑10‑01
- Replaced the BCH-Repetition decoder with RMRS decoder (strictly better).
- Released new parameter sets with updated sizes and timing:
    - hqc-128: KeyGen 136, Encaps 220, Decaps 384
    - hqc-192: KeyGen 305, Encaps 501, Decaps 821
    - hqc-256: KeyGen 545, Encaps 918, Decaps 1538
- All updates implemented in constant time.
- Hardware performance figures added.

### 2020‑05‑04
- Introduced improved error distribution analysis, reducing public key sizes by ~3%.
- Added a new decoding method combining Reed-Muller and Reed-Solomon codes, reducing public key size by ~17% for 128-bit security.
- Retained only one parameter set per security level (both HQC and HQC-RMRS).
- Optimized AVX2 implementations:
    - Now constant-time.
    - Avoid secret-dependent memory access.
    - Eliminated dependence on third-party libraries.
- Performance (AVX2, kilocycles):
    - HQC: KeyGen 175, Encaps 286, Decaps 486
    - HQC-RMRS: KeyGen 160, Encaps 272, Decaps 556

### 2019-04-10
- Added an optimized AVX2 C implementation exploiting low Hamming weight vectors.
- Added a constant-time BCH decoding implementation.
- Reference implementation initially relied on NTL.
