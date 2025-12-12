/**
 * @file api_chacekey.h
 * @brief APIs for operatons of cached keys
 */

#ifndef API_CACHEKEY_H
#define API_CACHEKEY_H

#include "stdint.h"

#include "parameters.h"

/// cached key structures ///

#include "gf2x.h"
// include for these macros:
// R_FFTFORM_BYTES
// R_FFTFORM_RLONLY_BYTES

#pragma pack(push, 1)

#include "data_structures.h"

typedef struct {
  unsigned char seed_ek[SEED_BYTES];
  unsigned char s[VEC_N_SIZE_BYTES];
} pk_t;

typedef struct {
  pk_t pk;
  unsigned char seed_dk[SEED_BYTES];
  unsigned char sigma[PARAM_SECURITY_BYTES];
  unsigned char seed_kem[SEED_BYTES];
} sk_t;



typedef struct {
  pk_t pk;
  unsigned char s_fft[R_FFTFORM_RLONLY_BYTES];
  unsigned char h_fft[R_FFTFORM_BYTES];
} cached_pk_t;

typedef struct {
  sk_t sk;
  unsigned char y_fft[R_FFTFORM_BYTES];
  unsigned char s_fft[R_FFTFORM_RLONLY_BYTES];
  unsigned char h_fft[R_FFTFORM_BYTES];
} cached_sk_t;

#pragma pack(pop)

#define CACHE_SECRETKEYBYTES               sizeof(cached_sk_t)
#define CACHE_PUBLICKEYBYTES               sizeof(cached_pk_t)

/// HQC with cached keys ///

#include "api.h"
// include for these macros:
// #define CRYPTO_SECRETKEYBYTES
// #define CRYPTO_PUBLICKEYBYTES
// #define CRYPTO_BYTES
// #define CRYPTO_CIPHERTEXTBYTES

/// @brief format transformation functions
/// @param cpk[out]: cached_pk_t cached public key
/// @param pk[in]: pk_t public key
void cache_hqcpke_pk( unsigned char * cpk , const unsigned char * pk );
/// @brief format transformation functions
/// @param csk[out]: cached_sk_t cached secret key
/// @param sk[in]: sk_t secret key
void cache_hqcpke_sk( unsigned char * csk , const unsigned char * sk );


/// HQC-PKE API ///

void cache_hqcpke_keygen(unsigned char* pk, unsigned char* sk, unsigned char * cpk, unsigned char* csk, uint8_t *seed);
void cache_hqcpke_encrypt(ciphertext_pke_t *c_pke, const uint8_t *cpk, const uint64_t *m, const uint8_t *theta);
uint8_t cache_hqcpke_decrypt(uint64_t *m, const uint8_t *csk, const ciphertext_pke_t *c_pke);

/// KEM API ///

int cache_hqckem_keypair(unsigned char* pk, unsigned char* sk , unsigned char *cpk, unsigned char* csk);
int cache_hqckem_enc(unsigned char* ct, unsigned char* ss, const unsigned char* cpk);
int cache_hqckem_dec(unsigned char* ss, const unsigned char* ct, const unsigned char* csk);

#endif
