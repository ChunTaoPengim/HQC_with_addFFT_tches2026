# `build_nist_release.sh`

> Assemble the NIST release directory under `build-package`


## Overview

This script automates creation of the `build-package` tree, ready for NIST PQC release. It will produce:

- **Reference_Implementation/**  
  ├─ `hqc-1/`  
  ├─ `hqc-3/`  
  └─ `hqc-5/`

- **Optimized_Implementation/**  
  ├─ `hqc-1/`  
  ├─ `hqc-3/`  
  └─ `hqc-5/`

- **KATs/**
    - Copies `.req`/`.rsp` files from `kats/ref/...` into `Reference_Implementation/{hqc-*}`
    - Copies from `kats/x86_64/...` into `Optimized_Implementation/{hqc-*}`

- **Supporting_Documentation/**
    - Empty placeholder directory for `HQC_Submission.pdf`

---

## Usage

From the project root:

```bash
chmod +x packaging/build_nist_release.sh
./packaging/build_nist_release.sh
